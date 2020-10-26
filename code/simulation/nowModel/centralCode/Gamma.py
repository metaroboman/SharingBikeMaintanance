import numpy as np
import random as random
import csv
import heapq
from tqdm import tqdm
from math import factorial
from multiprocessing import Pool
import time
import os
import shutil

from testParameters import A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_
from testODE import getStateDicts


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
    def __init__(self, portionState, A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_):     
        self.timeLimit = WARMTIME + RUNTIME
        self.areas = list(range(A))
        self.epi = 0
        self.portionState = portionState
        self.State = {}
        for s in list(portionState.keys()):
            self.State[s] = [0.0] * EPISODES
        self.Performance = [0] * EPISODES
        self.A, self.M, self.Pij, self.ArrLst, self.RhoMtx, self.Beta, self.B_, self.Gamma, self.Mu, self.N, self.Delta, self.D_ = \
            A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_
        self.nQueue = self.A*(self.A+1)+3
        
    def initState(self):
        if self.para != 0: tempState = random.choices(list(self.portionState.keys()), weights=self.portionState.values(), k=1)[0]
        else: tempState = [int(self.M/self.A)]*self.A + [0]*self.nQueue
        self.state1 = list(tempState[:self.A] + tempState[-3:])
        self.state2 = [list(tempState[(i+1)*self.A:(i+2)*self.A]) for i in range(self.A)]
        return self.state1, self.state2
    def initSchedu(self):
        for i in range(self.A):
            # arrival event at each area
            self.start = i
            self.addEvent(-1)
            # riding event of each riding queue
            for j in range(self.A):
                self.terminal = j
                [self.addEvent(1) for _ in range(self.state2[i][j])]
        # gather event at BP
        if self.state1[-3] >= self.B_: self.addEvent(2)
        self.state1[-2], broken = 0, self.state1[-2]
        for i in range(broken): 
            self.addEvent(3)
            self.state1[-2] += 1
        if self.state1[-1] >= self.D_: self.addEvent(4)
        
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
        self.stateRecord = self.state1[:self.A] + self.state2[0] + self.state2[1] + self.state1[-3:]
        return self.state1, self.state2, self.T
    
    def getRecord(self):
        tPortn = (self.T - self.formerT) / (WARMTIME + RUNTIME)
        self.normalBikes += (sum(self.state1[:A]) + sum([sum(_) for _ in self.state2])) * tPortn
        self.brokenBikes += sum(self.state1[A:]) * tPortn
        self.BP += self.state1[-3] * tPortn
        if self.state1[-2] == 0: self.idleRate += tPortn
        else: self.RC += self.state1[-2] * tPortn
        self.DP += self.state1[-1] * tPortn
        tempState2 = []
        for _ in self.state2: tempState2 += _
        tempState = tuple(self.state1[:-3] + tempState2 + self.state1[-3:])
        self.State[tempState][self.epi] += tPortn
        self.formerT = self.T

    def setPerformance(self, action):
        if action==1: self.arrivals += 1
        else: self.lostCustomers += 1
    
    def storeData(self):
        with open(FileAdd+'/stateDict' +NAME + str(self.para)+'.csv', 'w') as fout:
            writer = csv.writer(fout)
            for s in self.State:
                writer.writerow(list(s)+self.State[s])
        with open(FileAdd+'/systemPerformance' +NAME+str(self.para)+'.csv', 'w') as fout:
            writer = csv.writer(fout)
            for p in self.Performance:
                writer.writerow(p)
        
    def returnPerformance(self):
        re = []
        perf = np.array(self.Performance)
        for i in range(NUMBERPERF):
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
        #print(self.Performance)
        return self.returnPerformance()
    
    def addEvent(self, kind):
        if kind == -1:
            next_time = random.expovariate(self.ArrLst[self.start]) + self.T
            start, end = self.start, self.start
        elif kind == 1:
            next_time = random.expovariate(self.RhoMtx[self.start][self.terminal]) + self.T
            start, end = self.start, self.terminal
        elif kind == 2: 
            next_time = random.expovariate(self.Gamma)
            next_time += self.T
            start, end = 'b', 'f'
        elif kind == 3:
            next_time = random.expovariate(self.Mu) 
            if self.state1[-2] < self.N:
                next_time += self.T 
                heapq.heappush(self.F, next_time)
            else: 
                next_time += heapq.heappop(self.F)
                heapq.heappush(self.F, next_time)
            start, end = 'f', 'd'
        else: 
            next_time = random.expovariate(self.Delta)
            next_time += self.T
            start, end = 'd', 'ni'
        heapq.heappush(self.scheduler, [next_time, kind, start, end])
        
    def bikeArr(self):
        self.state2[self.start][self.terminal] -= 1
        if random.random()<self.Beta:
            self.state1[-3] += 1
            heapq.heappop(self.scheduler)
            if self.state1[-3] == self.B_:
                self.addEvent(2)
        else:
            self.state1[self.terminal] += 1
            heapq.heappop(self.scheduler)
    def BPover(self):
        heapq.heappop(self.scheduler)
        for i in range(self.B_): 
            self.state1[-3] -= 1
            self.addEvent(3) 
            self.state1[-2] += 1
        if self.state1[-3] >= self.B_: self.addEvent(2)
    def repair(self):
        heapq.heappop(self.scheduler)
        if self.state1[-2] <= self.N: heapq.heappop(self.F)
        self.state1[-2] -= 1
        self.state1[-1] += 1
        if self.state1[-1] == self.D_:
            self.addEvent(4)
    def DPover(self):
        heapq.heappop(self.scheduler)
        self.state1[-1] -= self.D_
        for i in range(self.A): self.state1[i] += self.D_/self.A
        if self.state1[-1] >= self.D_: self.addEvent(4)
    def cusArr(self):
        self.setPerformance(1)
        if self.state1[self.start] == 0:  # 但没车
            heapq.heappop(self.scheduler)
            self.addEvent(-1)
            self.setPerformance(-1)
        else:
            heapq.heappop(self.scheduler)
            self.addEvent(-1)
            # below use self.terminal to represent the target
            self.terminal = random.choices(self.areas, weights=self.Pij[self.start], k=1)[0]
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
    for i in range(STEPSIZE*p, STEPSIZE*(p+1)):
        Beta = i*0.5
        print(A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_)
        State, portionState = getStateDicts(A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_)
        env = Model(portionState, A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_)
        print('running', Beta)
        result.append(env.simulate(Beta))
    return result

def writeResult(re):
    with open(FileAdd+'/vary' +NAME+ '.csv', 'w') as fout:
        writer = csv.writer(fout)
        for i in range(CORE):
            for j in range(STEPSIZE):
                for k in range(NUMBERPERF):
                    writer.writerow([i*STEPSIZE+j, k]+re[i][j][k])
    
# Hyper parameters
EPISODES = 100 # times every setting is replicated
# random.seed(1)
WriteFile = False
#WriteFile = True
Name = 'Gamma'
FileAdd = 'C:/Rebalancing/nowModel/result/'+'A'+str(A)+'M'+str(M)+Name
WarmTime = 500
RunTime = 20000
Core = 4
NUMBERPERF = 7
STEPSIZE = 3
    
if __name__ == '__main__':
    re = []
    os.mkdir(FileAdd)
    
    start = time.time()
    p = Pool(Core)
    re += p.map(run, list(range(Core)))
    p.close()
    p.join()
    writeResult(re)
    end = time.time()
    print(end-start)
