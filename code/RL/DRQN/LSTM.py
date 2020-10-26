import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# np.random.seed(1)
# USE_CUDA = torch.cuda.is_available()


class DRQN(nn.Module):
    def __init__(self):
        self.A = 4
        super(DRQN, self).__init__()
        self.lstm_input_dimension = self.A * 2
        self.lstm = nn.LSTM(input_size=self.lstm_input_dimension, hidden_size=128, num_layers=1)
        self.fc1 = nn.Linear(128, 16)
        self.fc2 = nn.Linear(16, self.A)

    def forward(self, x, hidden):
        #x.view(1,-1,8)
        #print(x.size())
        x, new_hidden = self.lstm(x, hidden)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x[:,-1:].squeeze(dim=1), new_hidden