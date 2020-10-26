import numpy as np
import random as random
import heapq


    
class BikeModelS():
    '''
    This is the central model
    '''
    # initiate the parameters in this function
    def __init__(self, A, M, Pij, ArrLst, RhoMtx, Beta, Tau, C, Mu):     
        self.timeLimit = 25000
        self.areas = list(range(A))
        self.A, self.M, self.Pij, self.ArrLst, self.RhoMtx, self.Beta, self.Tau, self.C, self.Mu = \
            A, M, Pij, ArrLst, RhoMtx, Beta, Tau, C, Mu
        self.N = self.C[0][0]
        self.C = self.C[1:]
                
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
    def reset(self):
        self.T = 0 # time cursor
        self.formerT = 0
        # serve performance parameters
        self.normalBikes, self.brokenBikes, self.onBikes = self.M, 0, self.M
        # maintainance performance parameters
        self.idleRate, self.BP, self.RC, self.DP = 0, 0, 0, 0
        self.arrivals, self.lostCustomers = 0, 0
        # indicators of time stamp
        [self.nt, self.bt, self.it, self.bpt, self.rct, self.dpt, self.ot] = [0]*7
        self.enormalBikes, self.ebrokenBikes, self.eon= 0, 0, 0
        # maintainance performance parameters
        self.eidleRate, self.eBP, self.eRC, self.eDP = 0, 0, 0, 0
        self.Result = [0.0] * 8
        
        self.state1 = [int(self.M/self.A)]*self.A + [0]*3 # Nis, BP, FC, DP, OT, I
        self.state2 = [[0]*self.A for i in range(self.A)] # Rijs
        self.F = [] # time to be empty for fixing queue
        heapq.heapify(self.F)
        
        self.scheduler = []
        heapq.heapify(self.scheduler)
        for i in range(self.A):
            heapq.heappush(self.scheduler, [random.expovariate(self.ArrLst[i]), -1, i, i])
        for i, _ in enumerate(self.C):
            if _[-1] == 0: continue
            heapq.heappush(self.scheduler, [random.expovariate(_[1]*self.Tau), 2, _[0], _[1]])
        #self.stateRecord = self.state1[:A] + self.state2[0] + self.state2[1] + self.state1[-3:]
        
        # save state for drawing
        self.hn, self.hl, self.hi, self.ho = [[],[]], [[],[]], [[],[]], [[], []]
                
    def setRecord(self, kind):
        if kind == 1:
            self.enormalBikes = (self.enormalBikes * self.nt + self.normalBikes * (self.T - self.nt)) / self.T
            self.normalBikes, self.nt = self.normalBikes - 1, self.T
            self.hn[0].append(self.T)
            self.hn[1].append(self.enormalBikes/self.M)
        elif kind == 2:
            if self.RC == 0: 
                self.eidleRate = (self.eidleRate * self.it + (self.T - self.it)) / self.T
                self.hi[0].append(self.T)
                self.hi[1].append(self.eidleRate)
                self.it = self.T
            self.eRC = (self.eRC * self.rct + self.RC * (self.T - self.rct)) / self.T
            self.RC, self.rct = self.RC + min(self.start, self.state1[-3]), self.T
        elif kind == 3:
            self.eRC = (self.eRC * self.rct + self.RC * (self.T - self.rct)) / self.T
            self.RC, self.rct = self.RC - 1, self.T
            if self.RC == 0: 
                self.eidleRate = (self.eidleRate * self.it) / self.T
                self.hi[0].append(self.T)
                self.hi[1].append(self.eidleRate)
                self.it = self.T
        elif kind == 4: 
            self.enormalBikes = (self.enormalBikes * self.nt + self.normalBikes * (self.T - self.nt)) / self.T
            self.normalBikes, self.nt = self.normalBikes + min(self.start, self.state1[-1]), self.T
            self.hn[0].append(self.T)
            self.hn[1].append(self.enormalBikes/self.M)
        elif kind == -10:
            self.arrivals += 1
            self.hl[0].append(self.T)
            self.hl[1].append(self.lostCustomers/self.arrivals)
        elif kind == -11:
            self.lostCustomers += 1
            self.hl[0].append(self.T)
            self.hl[1].append(self.lostCustomers/self.arrivals)
        else: print('fucking wrong')
    
    def simulate(self):
        self.reset()
        while self.T <= self.timeLimit:
            self.stepForward()
        return np.mean(self.hn[1][-1000:]),np.mean(self.hl[1][-1000:]), np.mean(self.hi[1][-1000:])


    def addEvent(self, kind):
        if kind == -1:
            next_time = random.expovariate(self.ArrLst[self.start]) + self.T
            start, end = self.start, self.start
        elif kind == 1:
            next_time = random.expovariate(self.RhoMtx[self.start][self.terminal]) + self.T
            start, end = self.start, self.terminal
        elif kind == 2: 
            next_time = random.expovariate(self.terminal*self.Tau)
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
            next_time = random.expovariate(self.terminal*self.Tau)
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
        capa = self.start
        k = min(self.state1[-3], capa)
        for i in range(k): 
            self.addEvent(3) 
            self.state1[-3] -= 1
            self.state1[-2] += 1
        self.addEvent(4) # # of nb is 0 has been dealt before
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
        capa = self.start
        k = min(self.state1[-1], capa)
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
            self.terminal = random.choices(self.areas, weights=self.Pij[self.start], k=1)[0]
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

        #return self.state1, self.state2, self.T
