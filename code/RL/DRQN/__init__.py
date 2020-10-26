from LSTM import DRQN
import torch
import torch.nn as nn
#from model import Train
#from SimulationOptimization import BikeNet, Area, binaryInsert
import numpy as np
import pandas as pd
from collections import deque
import random

MEM_SIZE = 100
GAMMA = 0.99
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

class Memory():

    def __init__(self, memsize):
        self.memsize = memsize
        self.memory = deque(maxlen=self.memsize)

    def add_episode(self, epsiode):
        self.memory.append(epsiode)

    def get_batch(self, bsize):  # ,time_step):
        sampled_epsiodes = random.sample(self.memory, bsize)
        batch = []
        for episode in sampled_epsiodes:
            # point = np.random.randint(0,len(episode)+1-time_step)
            batch.append(episode)
        return batch


#mem = Memory(MEM_SIZE)
mem = []
def fill_memory():
    df = pd.read_csv('/Users/valarian/SJTU/SJTU/毕业论文/Data/data/2_reward_instances_4_80part1.csv', index_col=0)
    df.drop(labels='19', axis=1, inplace=True)
    for i in range(MEM_SIZE):
        bulk = df[df['0'] == i + 1].values
        s = bulk[:66, 1:9]
        a = bulk[:66, 9]
        r = bulk[:66, 18]
        s_ = bulk[:66, 10:18]
        #mem.add_episode([s, a, r, s_])
        mem.append([s, a, r, s_])

def train():
    eval_net = DRQN()
    tar_net = DRQN()
    tar_net.load_state_dict(eval_net.state_dict())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=0.00025)
    hidden_state = (torch.zeros(1,66,128).float().to(device), torch.zeros(1,66,128).float().to(device))

    step = 0
    for m in mem:
        if step%20 == 0:
            tar_net.load_state_dict(eval_net.state_dict())
        torch_current_states = torch.from_numpy(m[0]).float().view(1,-1,8).to(device)
        torch_acts = torch.from_numpy(m[1]).long().to(device)
        torch_rewards = torch.from_numpy(m[2]).float().to(device)
        torch_next_states = torch.from_numpy(m[3]).float().view(1,-1,8).to(device)

        Q_next, _ = tar_net.forward(torch_next_states, hidden_state)
        Q_next_max, __ = Q_next.detach().max(dim=1)
        target_values = torch_rewards[-1] + (GAMMA * Q_next_max)

        Q_s, _ = eval_net.forward(torch_current_states, hidden_state)
        Q_s_a = Q_s[0][torch_acts[-1]] #.gather(dim=1, index=torch_acts[-1].unsqueeze(dim=1)).squeeze(dim=1)

        loss = criterion(Q_s_a, target_values)

        #  save performance measure
        #loss_stat.append(loss.item())

        # make previous grad zero
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update params
        optimizer.step()

        step += 1

if __name__ == '__main__':
    random.seed(0)
    fill_memory()
    train()