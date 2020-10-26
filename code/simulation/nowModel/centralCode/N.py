import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import random as random
import csv
import heapq
from tqdm import tqdm
from math import factorial
from multiprocessing import Pool
import time
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
import os
from testODE import State, portionState

from testParameters import A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_

# set the parameters
# get the number of states S
'''
A: number of areas
M: number of total bikes
S: number of total states
Pij: transfering possibility matrix
Beta: broken rate
ArrMtx: arrival rates of each area
Gamma: gathering rate
Mu: fix rate
Delta: distributing rate
RhoMtx: matrix of ride rates
N: number of fix servers
B_: valve value at broken pool
D_: valve value at distributing pool
'''


class Model():
    '''
    This is the central model
    '''
    # initiate the parameters in this function
    def __init__(self):     
        self.timeLimit = WarmTime + RunTime
        self.areas = list(range(A))
        self.epi = 0
        self.State = {}
        for k,s in enumerate(State):
            self.State[s] = [0.0] * EPISODES
        self.Performance = [0] * EPISODES
        
    def initState(self):
        tempState = random.choices(list(portionState.keys()), weights=portionState.values(), k=1)[0]
        self.state1 = list(tempState[:A] + tempState[-3:])
        self.state2 = [list(tempState[(i+1)*A:(i+2)*A]) for i in range(A)]
        return self.state1, self.state2
    def initSchedu(self):
        for i in range(A):
            # arrival event at each area
            self.start = i
            self.addEvent(-1)
            # riding event of each riding queue
            for j in range(A):
                self.terminal = j
                [self.addEvent(1) for _ in range(self.state2[i][j])]
        # gather event at BP
        if self.state1[-3] >= B_: self.addEvent(2)
        self.state1[-2], broken = 0, self.state1[-2]
        for i in range(broken): 
            self.addEvent(3)
            self.state1[-2] += 1
        if self.state1[-1] >= D_: self.addEvent(4)
        
    def reset(self):
        self.T = 0 # time cursor

        self.scheduler = []
        heapq.heapify(self.scheduler)
        self.F = [] # time to be empty for fixing queue
        heapq.heapify(self.F)
        self.state1, self.state2 = self.initState()
        self.initSchedu()
        
        #for store performance
        # serve performance parameters
        self.normalBikes, self.brokenBikes = 0, 0
        # maintainance performance parameters
        self.idleRate, self.BP, self.RC, self.DP = 0, 0, 0, 0
        self.arrivals, self.lostCustomers = 0, 0
        self.formerT = 0
        self.stateRecord = self.state1[:A] + self.state2[0] + self.state2[1] + self.state1[-3:]
        return self.state1, self.state2, self.T
    
    def getRecord(self):
        tPortn = (self.T - self.formerT) / (WarmTime + RunTime)
        self.normalBikes += (sum(self.state1[:A]) + sum([sum(_) for _ in self.state2])) * tPortn
        self.brokenBikes += sum(self.state1[A:]) * tPortn
        self.BP += self.state1[-3] * tPortn
        if self.state1[-2] == 0: self.idleRate += tPortn
        else: self.RC += self.state1[-2] * tPortn
        self.DP += self.state1[-1] * tPortn
        tempState2 = []
        for _ in self.state2: tempState2 += _
        self.tempState = tuple(self.state1[:-3] + tempState2 + self.state1[-3:])
        self.State[self.tempState][self.epi] += tPortn
        self.formerT = self.T
        #return self.tempState + [tPortn] 

    def setPerformance(self, action):
        if action==1: self.arrivals += 1
        else: self.lostCustomers += 1
       
    def storeData(self):
        with open(FileAdd+'/stateDict' +Name + str(self.para)+'.csv', 'w') as fout:
            writer = csv.writer(fout)
            for s in self.State:
                writer.writerow(list(s)+self.State[s])
        with open(FileAdd+'/systemPerformance' +Name+str(self.para)+'.csv', 'w') as fout:
            writer = csv.writer(fout)
            for p in self.Performance:
                writer.writerow(p)
        
    def returnPerformance(self):
        re = []
        perf = np.array(self.Performance)
        for i in range(7):
            re.append([np.average(perf[:, i]), np.var(perf[:,i])**0.5])
        return re
        
    def simulate(self, para):
        self.para = para
        if WriteFile:
            with open(FileAdd, 'w') as fout:
                writer = csv.writer(fout)
                for i in range(EPISODES):
                    self.reset()
                    self.epi = i
                    while self.T <= self.timeLimit:
                        # print(self.T)
                        self.stepForward()
                        self.getRecord()
                        #writer.writerow(self.getRecord)  
        else:
            for i in range(EPISODES):
                self.epi = i
                self.reset()
                while self.T <= self.timeLimit:
                    self.stepForward()
                    self.getRecord()
                self.Performance[self.epi] = [self.normalBikes, self.brokenBikes, self.lostCustomers/self.arrivals, self.idleRate, self.BP, self.RC, self.DP]
        self.storeData()
        return self.returnPerformance()

    def addEvent(self, kind):
        if kind == -1:
            next_time = random.expovariate(ArrLst[self.start]) + self.T
            start, end = self.start, self.start
        elif kind == 1:
            next_time = random.expovariate(RhoMtx[self.start][self.terminal]) + self.T
            start, end = self.start, self.terminal
        elif kind == 2: 
            next_time = random.expovariate(Gamma)
            next_time += self.T
            start, end = 'b', 'f'
            #print('add event 2')
        elif kind == 3:
            next_time = random.expovariate(Mu) 
            if self.state1[-2] < N:
                next_time += self.T 
                heapq.heappush(self.F, next_time)
            else: 
                next_time += heapq.heappop(self.F)
                heapq.heappush(self.F, next_time)
            start, end = 'f', 'd'
        else: 
            next_time = random.expovariate(Delta)
            next_time += self.T
            start, end = 'd', 'ni'
        #if next_time<=RunTime+WarmTime:
        heapq.heappush(self.scheduler, [next_time, kind, start, end])
        
    def bikeArr(self):
        self.state2[self.start][self.terminal] -= 1
        if random.random()<Beta:
            self.state1[-3] += 1
            heapq.heappop(self.scheduler)
            if self.state1[-3] == B_:
                self.addEvent(2)
        else:
            self.state1[self.terminal] += 1
            heapq.heappop(self.scheduler)
    def BPover(self):
        heapq.heappop(self.scheduler)
        for i in range(B_): 
            self.state1[-3] -= 1
            self.addEvent(3) 
            self.state1[-2] += 1
        if self.state1[-3] >= B_: self.addEvent(2)
    def repair(self):
        heapq.heappop(self.scheduler)
        if self.state1[-2] <= N: heapq.heappop(self.F)
        self.state1[-2] -= 1
        self.state1[-1] += 1
        if self.state1[-1] == D_:
            self.addEvent(4)
    def DPover(self):
        heapq.heappop(self.scheduler)
        self.state1[-1] -= D_
        for i in range(A): self.state1[i] += D_/A
        if self.state1[-1] >= D_: self.addEvent(4)
    def cusArr(self):
        #print(self.state1, self.state2)
        #print('------------------------')
        self.setPerformance(1)
        if self.state1[self.start] == 0:  # 但没车
            heapq.heappop(self.scheduler)
            self.addEvent(-1)
            self.setPerformance(-1)
        else:
            heapq.heappop(self.scheduler)
            self.addEvent(-1)
            # below use self.terminal to represent the target
            self.terminal = random.choices(self.areas, weights=Pij[self.start], k=1)[0]
            self.state1[self.start] -= 1
            self.state2[self.start][self.terminal] += 1 
            self.addEvent(1)

    def stepForward(self):
        event = self.scheduler[0]
        #print(event)
        self.T, self.kind, self.start, self.terminal = event[0], event[1], event[2], event[3]
        '''
        kind of events:
        -1: customer ride a bike away
         1: a bike arrives at any area
         2: BP greater than B_
         3: a bike is fixed
         4: DP greater than D_
        '''
        if self.kind == 1: 
            self.bikeArr() # 顾客骑行到达
        elif self.kind == 2:
            self.BPover() # 坏车达到阈值
        elif self.kind == 3:
            self.repair() # 修好一辆车
        elif self.kind == 4:
            self.DPover() # 再分配
        else:# 顾客到达
            self.cusArr() #顾客到达

        return self.state1, self.state2, self.T


def run(p):
    result = []
    stage = 1
    for i in range(stage*p, stage*(p+1)):
        env = Model()
        N = para = i
        print('running', para)
        result.append(env.simulate(para))
    return result

def writeResult(re):
    with open(FileAdd+'/vary' +Name+ '.csv', 'w') as fout:
        writer = csv.writer(fout)
        stage = 1
        for i in range(Core):
            for j in range(stage):
                for k in range(7):
                    writer.writerow([i*stage+j, k]+re[i][j][k])
    
# Hyper parameters
EPISODES = 100 # times every setting is replicated
# random.seed(1)
WriteFile = False
#WriteFile = True
Name = 'N'
FileAdd = 'C:/Rebalancing/nowModel/result/'+'A'+str(A)+'M'+str(M)+Name
WarmTime = 500
RunTime = 4500
Core = 4
    
if __name__ == '__main__':
    re = []
    os.mkdir(FileAdd)
    
    start = time.time()
    p = Pool(Core)
    re += p.map(run, [1,2,3,4])
    p.close()
    p.join()
    writeResult(re)
    end = time.time()
    print(end-start)
