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
import requests

#from ParametersRS import A, M, Pij, ArrLst, RhoMtx, Beta, Tau, C, Mu, N, V, FileAdd
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

np.random.seed(0)
random.seed(0)

#A = 10
#M = 50
A, M = 10, 200
#A, M = 2, 6
FileAdd = 'C:\\Rebalancing\\2020-8-31\\result\\A'+str(A)+'M'+str(M)

def getPij(a):
    temp = np.log1p(np.random.rand(A,A))
    return (temp/sum(temp)).T
Pij = getPij(A)
#ArrLst = 6*(np.random.rand(A))
ArrLst = 6*(np.random.rand(A))
Beta = 0.3

Tau = 1
#C = 10
C = 3
#C = [[2,1], [3,2]] #[capacity, number]
Mu = 1
N = 1
V = 1

RhoMtx = [[1/(abs(j-i)+1) for i in range(A)] for j in range(A)]
# test the simulation length needed

# test the simulation length needed

# test the simulation length needed

class Model():
    '''
    This is the central model
    '''
    # initiate the parameters in this function
    def __init__(self, A, M, Pij, ArrLst, RhoMtx, Beta, Tau, C, Mu, N, V):     
        self.timeLimit = 25000
        #self.timeLimit = 10000
        self.areas = list(range(A))
        #self.epi = 0
        
        #self.Performance = [0] * EPISODES
        self.A, self.M, self.Pij, self.ArrLst, self.RhoMtx, self.Beta, self.Tau, self.C, self.Mu, self.N, self.V = \
            A, M, Pij, ArrLst, RhoMtx, Beta, Tau, C, Mu, N, V
        
        self.color = ['r']
        
        def get_target_number():
            A, M, ArrLst = self.A, self.M, self.ArrLst
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
        self.normalBikes, self.brokenBikes, self.onBikes = M, 0, M
        # maintainance performance parameters
        self.idleRate, self.BP, self.RC, self.DP = 0, 0, 0, 0
        self.arrivals, self.lostCustomers = 0, 0
        # indicators of time stamp
        [self.nt, self.bt, self.it, self.bpt, self.rct, self.dpt, self.ot] = [0]*7
        self.enormalBikes, self.ebrokenBikes, self.eon= 0, 0, 0
        # maintainance performance parameters
        self.eidleRate, self.eBP, self.eRC, self.eDP = 0, 0, 0, 0
        self.Result = [0.0] * 8
        
        self.state1 = [int(M/A)]*A + [0]*3 # Nis, BP, FC, DP, OT, I
        self.state2 = [[0]*A for i in range(A)] # Rijs
        self.F = [] # time to be empty for fixing queue
        heapq.heapify(self.F)
        
        self.scheduler = []
        heapq.heapify(self.scheduler)
        for i in range(A):
            heapq.heappush(self.scheduler, [random.expovariate(ArrLst[i]), -1, i, i])
        for i, _ in enumerate(range(self.V)):
            heapq.heappush(self.scheduler, [random.expovariate(self.Tau), 2, i, i])
        #heapq.heappush(self.scheduler, [random.expovariate(Tau), 4, i, i])
        self.stateRecord = self.state1[:A] + self.state2[0] + self.state2[1] + self.state1[-3:]
        
        # save state for drawing
        self.hn, self.hl, self.hi, self.ho = [[],[]], [[],[]], [[],[]], [[], []]
        
        return self.state1, self.state2, self.T
        
    def setRecord(self, kind):
        if kind == -10:
            self.arrivals += 1
            if self.T > 23000:
                self.hl[1].append(self.lostCustomers/self.arrivals)
        elif kind == -11:
            self.lostCustomers += 1
            if self.T > 23000:
                self.hl[1].append(self.lostCustomers/self.arrivals)
    
    def simulate(self):
        
        EPISODES = 1
        #s = time.time()
        for i in range(EPISODES):
            self.epi = i
            self.reset()
            #plt.ion()
            #plt.figure()
            while self.T <= self.timeLimit:
                #while self.T <= self.timeLimit:
                self.stepForward()

        return np.mean(self.hl[1]),np.std(self.hl[1])


    def addEvent(self, kind):
        if kind == -1:
            next_time = random.expovariate(self.ArrLst[self.start]) + self.T
            start, end = self.start, self.start
        elif kind == 1:
            next_time = random.expovariate(self.RhoMtx[self.start][self.terminal]) + self.T
            start, end = self.start, self.terminal
        elif kind == 2: 
            next_time = random.expovariate(self.Tau)
            next_time += self.T
            start, end = self.start, self.terminal
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
            next_time = random.expovariate(self.Tau)
            next_time += self.T
            start, end = self.start, self.terminal
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
        else:
            self.state1[self.terminal] += 1
        self.assertion()
    def BPover(self):
        heapq.heappop(self.scheduler)
        self.setRecord(2)
        k = min(self.state1[-3], self.C)
        [self.addEvent(3) for _ in range(k)]
        self.state1[-3] -= k
        self.state1[-2] += k
        self.addEvent(4) 
        self.assertion()
    def repair(self):
        heapq.heappop(self.scheduler)
        self.setRecord(3)
        if self.state1[-2] <= self.N: heapq.heappop(self.F)
        self.state1[-2] -= 1
        self.state1[-1] += 1
        self.assertion()
    def DPover(self):
        heapq.heappop(self.scheduler)
        self.setRecord(4)
        k = min(self.state1[-1], self.C)
        self.state1[-1] -= k
        dbar = k
        for i in self.arr_rank: 
            if self.state1[i]>=self.num_dis[i]: continue
            else: 
                alloc_n = min(self.num_dis[i]-self.state1[i], dbar)
                self.state1[i] += alloc_n
                dbar -= alloc_n
                if dbar == 0: break
        assert dbar == 0 
        self.addEvent(2)
        self.assertion()
    def cusArr(self):
        #print(self.state1, self.state2)
        #print('------------------------')
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
        self.assertion()

    def stepForward(self):
        event = self.scheduler[0]
        #print(event)
        self.T, self.kind, self.start, self.terminal = event[0], event[1], event[2], event[3]
        '''
        kind of events:
        -1: customer ride a bike away
         1: a bike arrives at any area
         2: Carrier arrives at DP
         3: a bike is fixed
         4: Carrier arrives at BP
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

    REPEAT = 50
    #np.random.seed(0)
    #random.seed(0)
    Total = 300
    cr,cc = 2, 25
    for epi in tqdm(range(REPEAT)):
        #def getPij(a):
        #    temp = np.random.rand(A,A)
        #    return (temp/sum(temp)).T
        #Pij = getPij(A)
        #ArrLst = np.random.rand(A)
        #print(Pij, ArrLst)
        
        for smt in np.arange(3*p+1, 3*(p+1)+1,1):
        #for i in range(1):
            #smt = 0.2
            #smt = round(smt, 1)
            if smt > 11: continue
            N = smt
            para = V = (Total - cc*N)//cr
            instance = Model(A, M, Pij, ArrLst, RhoMtx, Beta, Tau, C, Mu, N, V)
            return_perf = list(instance.simulate())
            return_perf += [epi, smt, para]
            result.append(return_perf)
            del instance, return_perf

    return result
#print(run(0))
def writeResult(re):
    df = pd.DataFrame(re, columns=['l','ls','epi','smt','para'])
    df.to_csv(FileAdd+'TauRSMulticar3T300Part2.csv')
    
# Hyper parameters
EPISODES = 30 # times every setting is replicated
random.seed(1)

CORE = 3
NUMBERPERF = 7
STEPSIZE =4

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

    def send_notice(event_name, key, text):
        url = "https://maker.ifttt.com/trigger/"+event_name+"/with/key/"+key
        response = requests.request("POST", url)

        print(response.text)

    event_name = 'python_finished_running'
    key = 'bJK-bEYNYyBPXFnvNZ4P1x'
    text = '983'
    send_notice(event_name, key, text)
