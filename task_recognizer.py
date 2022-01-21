import torch
import torch.nn as nn
import numpy

data = torch.ones(5, 1, 3, 500, 500) # [batchsize, seqsize, channle, 500, 500]

class CNN(nn.Module):
    def __init__(self):
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

for i in range(5):
    pred = cnn(data[i])
    pred, h = lstm(pred, h)
    pred = torch.argmax(pred, dim=1)
    pred = pred.detach().numpy()
    h.detach()
    # pred = torch.stack(pred, dim=0)
    print()

print(CNN)
print(LSTM)
print('aaa ?')
print('aaa ?')
