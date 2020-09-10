import torch
import numpy as np
import random
random.seed(3)

class Feedforward(torch.nn.Module):

        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.dp1 = torch.nn.Dropout(p = 0.5)
            self.relu1 = torch.nn.ReLU()
            
            self.rnn1 = torch.nn.RNNCell(self.hidden_size, self.hidden_size)
            self.tanh1 = torch.nn.Tanh()
            self.dp2 = torch.nn.Dropout(p = 0.5)

            self.fc2 = torch.nn.Linear(self.hidden_size, 2)
            self.sigmoid = torch.nn.Sigmoid()        
        
        def forward(self, x):
            fc1 = self.fc1(x)
            dp1 = self.dp1(fc1)
            relu1 = self.relu1(dp1)
            
            rnn1 = self.rnn1(relu1)
            tanh1 = self.tanh1(rnn1)
            dp2 = self.dp2(tanh1)

            output = self.fc2(dp2)
            output = self.sigmoid(output)
            return output
