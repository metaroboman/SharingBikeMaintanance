import numpy as np
import pandas as pd
import random as random
import csv
import heapq
from tqdm import tqdm
from math import factorial
from multiprocessing import Pool
import time
import os
import shutil

from testParameters import A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_, FileAdd
#from testODE import getStateDicts


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


# test the simulation length needed

class Model():
    '''
    This is the central model
    '''
    # initiate the parameters in this function
    def __init__(self, A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_):     
        self.timeLimit = 25000
        self.areas = list(range(A))
        #self.epi = 0
        
        #self.Performance = [0] * EPISODES
        self.A, self.M, self.Pij, self.ArrLst, self.RhoMtx, self.Beta, self.B_, self.Gamma, self.Mu, self.N, self.Delta, self.D_ = \
            A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_
        
        self.color = ['r']
        
        def get_target_number():
            A, M, D_, ArrLst = self.A, self.M, self.D_, self.ArrLst
            arr = np.array(ArrLst)/sum(ArrLst)
            num_dis = [int(M*x) for x in arr]
            left = M - sum(num_dis)
            for i in np.argsort(ArrLst)[:left]:
                num_dis[i] += 1
            return num_dis
        self.num_dis = get_target_number()
        self.arr_rank = np.argsort(ArrLst)[::-1]
        #print(self.num_dis, self.ArrLst, self.arr_rank)
    def reset(self):
        self.T = 0 # time cursor
        self.formerT = 0
        
        # serve performance parameters
        self.normalBikes, self.brokenBikes = M, 0
        # maintainance performance parameters
        self.idleRate, self.BP, self.RC, self.DP = 0, 0, 0, 0
        self.arrivals, self.lostCustomers = 0, 0
        # indicators of time stamp
        [self.nt, self.bt, self.it, self.bpt, self.rct, self.dpt] = [0]*6
        self.enormalBikes, self.ebrokenBikes = 0, 0
        # maintainance performance parameters
        self.eidleRate, self.eBP, self.eRC, self.eDP = 0, 0, 0, 0
        self.Result = [0.0] * 7
        
        self.state1 = [int(M/A)]*A + [0]*3 # Nis, BP, FC, and DP
        self.state2 = [[0]*A for i in range(A)] # Rijs
        self.F = [] # time to be empty for fixing queue
        heapq.heapify(self.F)
        
        self.scheduler = []
        heapq.heapify(self.scheduler)
        for i in range(A):
            heapq.heappush(self.scheduler, [random.expovariate(ArrLst[i]), -1, i, i])
        self.stateRecord = self.state1[:A] + self.state2[0] + self.state2[1] + self.state1[-3:]
        
        # save state for drawing
        self.hn, self.hl, self.hi = [], [], [0]
        
        return self.state1, self.state2, self.T
        
    def setRecord(self, kind):
        if kind == 1:
            self.enormalBikes = (self.enormalBikes * self.nt + self.normalBikes * (self.T - self.nt)) / self.T
            self.normalBikes, self.nt = self.normalBikes - 1, self.T
            self.eBP = (self.eBP * self.bpt + self.BP * (self.T - self.bpt)) / self.T
            self.BP, self.bpt = self.BP + 1, self.T
            if self.T > 20000:
                self.hn.append(self.enormalBikes/self.M)
        elif kind == 2:
            if self.RC == 0: 
                self.eidleRate = (self.eidleRate * self.it + (self.T - self.it)) / self.T
                if self.T > 20000:
                    self.hi.append(self.eidleRate)
                self.it = self.T
            self.eBP = (self.eBP * self.bpt + self.BP * (self.T - self.bpt)) / self.T
            self.BP, self.bpt = self.BP - min(self.B_, self.state1[-3]), self.T
            self.eRC = (self.eRC * self.rct + self.RC * (self.T - self.rct)) / self.T
            self.RC, self.rct = self.RC + min(self.B_, self.state1[-3]), self.T
        elif kind == 3:
            self.eRC = (self.eRC * self.rct + self.RC * (self.T - self.rct)) / self.T
            self.RC, self.rct = self.RC - 1, self.T
            self.eDP = (self.eDP * self.dpt + self.DP * (self.T - self.dpt)) / self.T
            self.DP, self.dpt = self.DP + 1, self.T
            if self.RC == 0: 
                self.eidleRate = (self.eidleRate * self.it) / self.T
                if self.T > 20000:
                    self.hi.append(self.eidleRate)
                self.it = self.T
        elif kind == 4: 
            self.enormalBikes = (self.enormalBikes * self.nt + self.normalBikes * (self.T - self.nt)) / self.T
            self.normalBikes, self.nt = self.normalBikes + min(self.D_, self.state1[-1]), self.T
            self.eDP = (self.eDP * self.dpt + self.DP * (self.T - self.dpt)) / self.T
            self.DP, self.dpt = self.DP - min(self.D_, self.state1[-1]), self.T
            if self.T > 20000:
                self.hn.append(self.enormalBikes/self.M)
        elif kind == -10:
            self.arrivals += 1
            if self.T > 20000:
                self.hl.append(self.lostCustomers/self.arrivals)
        elif kind == -11:
            self.lostCustomers += 1
            if self.T > 20000:
                self.hl.append(self.lostCustomers/self.arrivals)
        else: print('fucking wrong')
    
    def simulate(self):
        self.reset()
        #plt.ion()
        while self.T <= self.timeLimit:
            self.stepForward()
        return np.mean(self.hn), np.mean(self.hl), np.mean(self.hi)

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
            if self.state1[-3] >= self.B_: B = self.B_
            else: B = self.state1[-3]
            #B = self.state1[-3]
            start, end = B, 'f'
            #print('add event 2')
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
            if self.state1[-1] >= self.D_: D = self.D_
            else: D = self.state1[-1]
            #D = self.state1[-1]
            start, end = D, 'ni'
        heapq.heappush(self.scheduler, [next_time, kind, start, end])
    
    def assertion(self):
        su = 0
        S = list(self.state1)
        S.extend(list(np.array(self.state2).reshape(-1)))
        for s in S:
            assert(s>=0 and s<=self.M)
            su += s
        assert su == self.M
    
    def bikeArr(self):
        self.state2[self.start][self.terminal] -= 1
        heapq.heappop(self.scheduler)
        if random.random()<self.Beta:
            self.state1[-3] += 1
            self.setRecord(1)
            if self.state1[-3] == self.B_ or self.normalBikes==0:
                # here above the conditions are important:
                # for the first: it's ==, so when there are more broken bikes
                # for the second: when the # of nb is 0
                # so there won't be more broken bikes
                self.addEvent(2)
        else:
            self.state1[self.terminal] += 1
    def BPover(self):
        heapq.heappop(self.scheduler)
        self.setRecord(2)
        for i in range(self.B_): 
            if self.state1[-3]>=1:
                self.state1[-3] -= 1
                self.addEvent(3) 
                self.state1[-2] += 1
        if self.state1[-3] >= self.B_: self.addEvent(2) # # of nb is 0 has been dealt before
    def repair(self):
        heapq.heappop(self.scheduler)
        if self.state1[-2] <= self.N: heapq.heappop(self.F)
        self.state1[-2] -= 1
        self.state1[-1] += 1
        self.setRecord(3)
        if self.state1[-1] == self.D_ or self.normalBikes==0:
            self.addEvent(4)
    def DPover(self):
        heapq.heappop(self.scheduler)
        self.setRecord(4)
        dbar = min(self.state1[-1], self.D_)
        for i in self.arr_rank: 
            if self.state1[i]>=self.num_dis[i]: continue
            else: 
                alloc_n = min(self.num_dis[i]-self.state1[i], dbar)
                self.state1[i] += alloc_n
                self.state1[-1] -= alloc_n
                dbar -= alloc_n
                if dbar == 0: break
        if self.state1[-1] >= self.D_: self.addEvent(4)
    def cusArr(self):
        self.setRecord(-10)
        if self.state1[self.start] == 0:  # 但没车
            heapq.heappop(self.scheduler)
            self.addEvent(-1)
            self.setRecord(-11)
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
    #print('run')
    result = []

    REPEAT = 2

    for epi in tqdm(range(REPEAT)):
        np.random.seed(epi)
        def getPij(a):
            temp = np.random.rand(A,A)
            return (temp/sum(temp)).T
        Pij = getPij(A)
        ArrLst = np.random.rand(A)
        for smt in range(STEPSIZE*p+2, STEPSIZE*(p+1)+2):
            #smt = round(smt, 1)
            for i in np.arange(0.1,smt,1):
                para = Gamma = i
                Delta = smt - i
                instance = Model(A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_)
                return_perf = list(instance.simulate())
                return_perf += [epi, smt, para]
                result.append(return_perf)
                del instance, return_perf

    return result

def writeResult(re):
    df = pd.DataFrame(re, columns=['n','l','i','epi','smt','para'])
    df.to_csv(FileAdd+'limitedSum.csv')
    
# Hyper parameters
EPISODES = 30 # times every setting is replicated
# random.seed(1)
WriteFile = False
#WriteFile = True
NAME = 'Beta'
WARMTIME = 0
#RUNTIME = 80000
CORE = 2
NUMBERPERF = 7
STEPSIZE =2

#run(3)
if __name__ == '__main__':
    re = []
    if os.path.exists(FileAdd):
        shutil.rmtree(FileAdd)
    os.mkdir(FileAdd)
    start = time.time()
    p = Pool(CORE)
    [re.extend(r) for r in list(p.map(run, list(range(CORE))))]
    #run(3)
    p.close()
    p.join()
    writeResult(re)
    end = time.time()
    print(end-start)
