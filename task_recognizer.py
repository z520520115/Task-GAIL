import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy

# data path
data_path = "./data/task_sample_videos"
save_model_path = "./crnn_model/"

# select which frame to begin & end in videos
b_frame, e_frame, s_frame = 1, 50, 5

# fake data for test
# data = torch.ones(5, 1, 3, 500, 500) # [batchsize, seqsize, channle, 500, 500]

class CNN(nn.Module):
    def __init__(self, kernel_size = (3,3),):
        super(CNN, self).__init__()
        # data shape = (1, 3, 500, 500)
        self.encoder = torch.nn.Sequential(
            # Conv1  = (1, 32, 498, 498) (500-3+0)/1+1 = 498
            # Pool1  = (1, 32, 166, 166)
            torch.nn.Conv2d(3, 32, kernel_size = (3,3), stride = (1,1), padding='valid'),
            torch.nn.AvgPool2d(kernel_size =(3,3), stride=(3,3)),
            # Conv2  = (1, 64, 164, 164)
            # Pool2  = (1, 64, 54, 54)
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid'),
            torch.nn.AvgPool2d(kernel_size=(3, 3)),
            # Conv3  = (1, 128, 52, 52)
            # Pool3  = (1, 128, 17, 17)
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid'),
            torch.nn.AvgPool2d(kernel_size=(3, 3)),
            # Conv4  = (1, 256, 15, 15)
            # Pool4  = (1, 256, 5, 5)
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding='valid'),
            torch.nn.AvgPool2d(kernel_size=(3, 3)),
            # Conv4  = (1, 512, 5, 5)
            # Pool4  = (1, 512, 1, 1)
            # Flat  = (1, 512)
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding='valid'),
            torch.nn.AvgPool2d(kernel_size=(3, 3)),
            torch.nn.Flatten()
        )

    def forward(self, x):
        x = self.encoder(x) # (1, 512)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size = 1536, hidden_size = 1024, output_size = 1024):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.i2o = nn.Linear(input_size, output_size)
        # self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        x_combined = torch.cat((x, hidden), 1)
        hidden = self.i2h(x_combined)
        x = self.i2o(x_combined)
        x = self.softmax(x)
        return x, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

cnn = CNN()
lstm = LSTM()

h = torch.randn(1, 1024)

for epoch in range(5):
    pred = cnn(data[i])
    pred, h = lstm(pred, h)
    pred = torch.argmax(pred, dim=1)
    pred = pred.detach().numpy()
    h.detach()
    # pred = torch.stack(pred, dim=0)
    print()

def train(log_intreval, model, device, train_loader, optimizer, epoch):
    # Set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train() # cnn
    rnn_decoder.train() # lstm

    losses = []
    scores = []
    N_count = 0 # counting total training sample in one epoch

    for batch_idx, (x, y) in enumerate(train_loader): # enumerate()将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
        # distribute data to device
        x,y = x.to(device), y.to(device).view(-1, ) #view 重构张量维度,数据一样形状不同,-1让电脑计算

        N_count += x.size(0)

        optimizer.zero_grad() # 梯度归零
        output = rnn_decoder(cnn_encoder(x)) # output dim = (batch_size, number of classes)

        # 计算交叉熵损失, output是网络输出的概率向量, y是真实标签(标量),output是没有经过softmax的,通过log_softmax和nll_loss来计算
        loss = F.cross_entropy(output, y)
        losses.append(loss.item()) # .item()返回浮点型,精度增加到小数点后16位

        # to compute accurary
        y_pred = torch.max(output, 1)[1] # 1是索引的维度,每一行的最大,[1]为返回最大值的每个索引
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), # .data()只返回variable中的数据部分(去掉variable containing)
                                    y_pred.cpu().data.squeeze().numpy()) # .numpy()将数据转化为numpy, .squeeze()将数据中维度为1的删除
        scores.append(step_score) # computed on GPU

        loss.backward() # 反向传播得到每个参数的梯度值
        optimizer.step() # 梯度下降执行一步参数更新

        # show infotmation
        if(batch_idx + 1) % log_intreval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))
        return losses, scores

def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval() # 指定eval时, 框架会自动固定BN和DropOut,不会取平均会用训练好的值
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad(): # 被该语句warp起来的部分将不会track梯度
        for x, y in test_loader:
            # distribute data to device
            x, y = x.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(x))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                  # sum up the batch loss
            y_pred = output.max(1, keepdim = True)[1] # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y) # extend()扩展list,将y的每个元素添加到all_y中
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim = 0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100 * test_score))

    # save Pytorch model of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_epoch{}.pth'.format(epoch + 1)))
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'lstm_epoch{.pth'.format(epoch + 1)))
    torch.save(optimizer.state.dict(), os.path.join(save_model_path, 'optimizer_eopch{}.pth'.format(epoch + 1)))
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score

# Detect devices
use_cuda = torch.cuda.is_available()                 # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu") # use CPU or GPU

# Data loading parameters
params = {'batch_size': 5, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}