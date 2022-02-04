import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import os
import numpy as np

# data path
data_path = "./data/task04/"
save_model_path = "./crnn_model/"

# cnn architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512
img_x, img_y = 90, 120
dropout_p = 0.0

# lstm architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 4                   # number of target category
epochs = 120
batch_size = 2
learning_rate = 1e-4
log_interval = 10       # interval for displaying training info

# select which frame to begin & end in videos
b_frame, e_frame, s_frame = 1, 29, 1

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

class CNN(nn.Module):
    def __init__(self, img_x=90, img_y=120, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(CNN, self).__init__()
        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), (0, 0), (5, 5), (2, 2))  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, (0, 0), (3, 3), (2, 2))
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, (0, 0), (3, 3), (2, 2))
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, (0, 0), (3, 3), (2, 2))

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        #     nn.BatchNorm2d(64, momentum=0.01),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=2),
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        #     nn.BatchNorm2d(128, momentum=0.01),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=2),
        # )
        #
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        #     nn.BatchNorm2d(256, momentum=0.01),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=2),
        # )

        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        # self.fc1 = nn.Linear(256 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc_hidden1)  # fully connected layer, output k classes
        self.fc1 = nn.Linear(256 * 4 * 6, self.fc_hidden1)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.conv1(x_3d[:, t, :, :, :])
            # x = self.conv2(x)
            # x = self.conv3(x)
            # x = self.conv4(x)
            x = x.view(x.size(0), -1)  # flatten the output of conv
            # x = x.view(-1, 4*6*256) # x.view 的第二个参数和nn.linear第一个参数一致
            # print(self.conv1_outshape, self.conv2_outshape, self.conv3_outshape, self.conv4_outshape)

            # FC layers
            x = F.relu(self.fc1(x)) # 256 * 4 * 6
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq

class LSTM(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(LSTM, self).__init__()
        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        x = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:d}.png'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            x.append(image)
        x = torch.stack(x, dim=0)

        return x

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        x = self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(x.shape)
        return x, y

def train_func(log_intreval, model, device, train_loader, optimizer, epoch):
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
    all_y = torch.stack(all_y, dim=0) # 拼接张量, 将all_y变成0维的新张量
    all_y_pred = torch.stack(all_y_pred, dim = 0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100 * test_score))

    # save Pytorch model of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_epoch{}.pth'.format(epoch + 1)))
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'lstm_epoch{}.pth'.format(epoch + 1)))
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_eopch{}.pth'.format(epoch + 1)))
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score

# Detect devices
use_cuda = torch.cuda.is_available()                 # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu") # use CPU or GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Data loading parameters
params = {'batch_size': 2, 'shuffle': True, 'pin_memory': True} if use_cuda else {}

# load Task actions names
task_labels = ['JumpForward', 'Run', 'TurnLeft', 'TurnRight']

# convert labels -> category
le = LabelEncoder()
le.fit(task_labels)

# show how many classes there are
list(le.classes_)

# convert category -> one-hot
action_category = le.transform(task_labels).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

tasks = []
fnames = os.listdir(data_path)

all_tasks_names = []
for f in fnames:
    loc1 = f.find('v_')
    loc2 = f.find('_g')
    tasks.append(f[(loc1 + 2) : loc2])

    all_tasks_names.append(f)

# all data files
x_list = all_tasks_names              # all video file names
y_list = le.transform(tasks)          # all video labels


# random split sample set to training set and test set
train_list, test_list, train_label, test_label = train_test_split(x_list, y_list, test_size=0.25, random_state=42, stratify=y_list)

transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(b_frame, e_frame, s_frame).tolist()

train_set, valid_set = Dataset_CRNN(data_path, train_list, train_label, selected_frames, transform=transform) ,\
                       Dataset_CRNN(data_path, test_list, test_label, selected_frames, transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

# create model
cnn = CNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
          drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
lstm = LSTM(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
            h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)
print(cnn)
print(lstm)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn)
    rnn_decoder = nn.DataParallel(lstm)

crnn_params = list(cnn.parameters()) + list(lstm.parameters())
optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train_func(log_interval, [cnn, lstm], device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = validation([cnn, lstm], device, optimizer, valid_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A, B, C, D = np.array(epoch_train_losses), np.array(epoch_train_scores), \
                 np.array(epoch_test_losses), np.array(epoch_test_scores)
    np.save('./outputs/CRNN_epoch_training_losses.npy', A)
    np.save('./outputs/CRNN_epoch_training_scores.npy', B)
    np.save('./outputs/CRNN_epoch_test_loss.npy', C)
    np.save('./outputs/CRNN_epoch_test_score.npy', D)

# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
title = "./result/crnn.png"
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()

