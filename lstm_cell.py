import torch
import torch.nn as nn
import numpy

class LSTM(nn.Module):
    def __init__(self, chord_vocab_size, key_vocab_size, chord_length, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.chord_embedding = nn.Embedding(chord_vocab_size, hidden_size)
        self.key_embedding = nn.Embedding(key_vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(chord_length, hidden_size)

        self.i2h = nn.Linear(hidden_size*4, hidden_size)
        self.i2o = nn.Linear(hidden_size*4, hidden_size)
        self.o2o = nn.Linear(hidden_size*2, hidden_size)
        self.o2l = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, key, pos, chord, hidden):
        key = self.key_embedding(key)
        chord = self.chord_embedding(chord)
        pos = self.pos_embedding(pos)
        input_combined = torch.cat((key, pos, chord, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.o2l(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)