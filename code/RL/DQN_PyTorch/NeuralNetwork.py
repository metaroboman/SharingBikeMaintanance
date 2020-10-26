"""
This part of code is the Deep Q Network (DQN) brain.
view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# np.random.seed(1)
# USE_CUDA = torch.cuda.is_available()


# Deep Q Network off-policy
class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQN, self).__init__()
        self.line1 = nn.Linear(num_inputs, 64)
        self.line2 = nn.Linear(64, 32)
        self.line3 = nn.Linear(32, num_outputs)

    def forward(self, x):
        x = F.sigmoid(self.line1(x))
        x = F.relu(self.line2(x))
        x = self.line3(x)
        return x



