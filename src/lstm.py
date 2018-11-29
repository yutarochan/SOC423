'''
SOC 423 - Birth Rate Prediction Model [LSTM]
Source: https://github.com/pytorch/examples/blob/master/time_sequence_prediction/train.py
Author: Yuya Ong
'''
from __future__ import print_function
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Application Parameters
SEED = 9892
PROJ = 1
PROJ_TEST = 20
EPOCH = 15

# Initialize Seed Values
np.random.seed(SEED)
torch.manual_seed(SEED)

class Model(nn.Module):
    def __init__(self, hid_size = 51):
        super(Model, self).__init__()

        # Define Model Parameters
        self.hid_size = hid_size

        # Define Model Objects
        self.lstm_1 = nn.LSTMCell(1, self.hid_size)
        self.lstm_2 = nn.LSTMCell(self.hid_size, self.hid_size)
        self.linear = nn.Linear(self.hid_size, 1)

    def forward(self, input, future = 0):
        outputs = []

        # Initialize LSTM Parameters
        h_t = torch.zeros(input.size(0), self.hid_size, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.hid_size, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.hid_size, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.hid_size, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.shape[1], dim=1)):
            h_t, c_t = self.lstm_1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm_2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):
            h_t, c_t = self.lstm_1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm_2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[:, -1].unsqueeze(0)

if __name__ == '__main__':
    # Load Dataset
    birth_rate = pd.read_csv('../data/raw/birth_rate.csv', index_col=1)
    birth_rate = birth_rate.drop(['Country Name', 'Indicator Name', 'Indicator Code', '2017', 'Unnamed: 62'], axis=1)
    birth_rate = birth_rate.dropna().T

    # Country Train/Test Split
    country_list = list(birth_rate)
    random.shuffle(country_list)
    train_set = country_list[:int(len(country_list)*0.75)]
    test_set = country_list[int(len(country_list)*0.75):]

    # Temporal Data Split
    train_X = torch.from_numpy(birth_rate[train_set].values[:-PROJ, :]).transpose(1, 0)
    train_Y = torch.from_numpy(birth_rate[train_set].values[train_X.shape[1]-PROJ+1:, :]).transpose(1, 0)
    test_X = torch.from_numpy(birth_rate[test_set].values[:-PROJ_TEST, :]).transpose(1, 0)
    test_Y = torch.from_numpy(birth_rate[test_set].values[test_X.shape[1]-PROJ_TEST+1:, :]).transpose(1, 0)

    # Setup Model
    model = Model()
    model.double()

    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    # Model Training Process
    for i in range(EPOCH):
        print('EPOCH: ' + str(i))

        def closure():
            optimizer.zero_grad()
            loss_total = 0
            for idx in range(train_X.shape[0]):
                out = model(train_X[idx].unsqueeze(0))
                loss = criterion(out, train_Y[idx].unsqueeze(0))
                loss_total += loss.item()
                loss.backward()
            print('> LOSS: ' + str(loss_total))
            return loss_total

        optimizer.step(closure)

    '''
    # Model Testing
    with torch.no_grad():
        loss_total = 0
        pred = seq(, future=PROJ_TEST)
        loss = criterion(pred[:, :-future], test_target)
        print('test loss:', loss.item())
        y = pred.detach().numpy()
    '''
