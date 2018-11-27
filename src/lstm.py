'''
SOC 423 - Birth Rate Prediction Model [LSTM]
Author: Yuya Ong
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

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

        # Iterate through Data
        for i, input_t, in enumerate(input.chunk(input_size(1), dim=1)):
            h_t, c_t = self.lstm_1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm_2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        # Generate Predictions
        for i in range(future):
            h_t, c_t = self.lstm_1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm_2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs
