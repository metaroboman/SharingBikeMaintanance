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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from math import factorial, isclose
from tqdm import tqdm
import csv
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve, dsolve, isolve
from collections import deque
import seaborn as sns
import random
import requests

from ParametersOde import A, M, Pij, ArrLst, RhoMtx, Beta, Tau, C, Mu, N, FileAdd
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
#Instance = 4
#np.random.seed(Instance)
#random.seed(0)
#A, M = 2, 6
#FileAdd = 'C:\\Rebalancing\\2020-8-31\\result\\A'+str(A)+'M'+str(M)
#def getPij(a):
#    temp = np.log1p(np.random.rand(A,A))
#    return (temp/sum(temp)).T
#Pij = getPij(A)
#ArrLst = 6*np.random.rand(A)
#Beta = 0.3
#Tau = 1.0
#C = 3
#Mu = 1
#N = 1
#RhoMtx = [[1/(abs(j-i)+1) for i in range(A)] for j in range(A)]

# test the simulation length needed

class Model():
    def __init__(self, A, M, Pij, ArrLst, RhoMtx, Beta, Tau, C, Mu, N,):
        # init parameters
        self.A, self.M, self.Pij, self.ArrLst, self.RhoMtx, self.Beta, self.Tau, self.C, self.Mu, self.N = \
            A, M, Pij, ArrLst, RhoMtx, Beta, Tau, C, Mu, N
        # init state dict
        self.State = self.getStateDict()
        self.n_state = len(self.State)

    def getStateDict(self):
        # generate state dictionary direrctly
        def getState(a, m):
            A, M = a, m
            State = {}
            index = [0]
            def temp(x, i, index):
                x.append(i)
                generate_R(x.copy(), index)
            def generate_R(s, index):
                if len(s) == A + 3 + A ** 2 and sum(s) == M:
                    State[tuple(s)] = index[0]
                    index[0] += 1
                elif len(s) > A + 3 + A ** 2 or sum(s) > M:
                    return 0
                else:
                    for i in range(M + 1):
                        temp(s.copy(), i, index)
            for i in range(M + 1):
                generate_R([i], index)
            return State
        State = getState(self.A, self.M)
        # print('getState done!')
        return State

    def assertion(self, s):
        assert sum(s) == self.M
        for i in s:
            assert i >= 0
            assert i <= M

    def getMtx(self):
        State = self.State
        n_state = self.n_state
        
        def get_target_number():
            A, M, C, ArrLst = self.A, self.M, self.C, self.ArrLst
            arr = np.array(ArrLst) / sum(ArrLst)
            num_dis = [int(M * x) for x in arr]
            left = M - sum(num_dis)
            for i in np.argsort(ArrLst)[::-1][:left]:
                num_dis[i] += 1
            assert sum(num_dis) == self.M
            return num_dis
        self.num_dis = get_target_number()
        
        def get_before_reallocate(arr_rank, num_dis, N_lst, dbar, result_lst):
            # if dbar == 0: return result_lst
            # print(arr_rank, num_dis, N_lst, dbar, result_lst)
            i = arr_rank[0]
            if N_lst[i] < num_dis[i]:
                if dbar <= N_lst[i]:
                    result = N_lst.copy()
                    result[i] -= dbar
                    result_lst.append(result)
            elif N_lst[i] > num_dis[i]:
                if arr_rank.size != 0:
                    get_before_reallocate(arr_rank[1:], num_dis, N_lst, dbar, result_lst)
            else:
                if arr_rank.size == 1:
                    if N_lst[i] >= dbar:
                        result = N_lst.copy()
                        result[i] -= dbar
                        result_lst.append(result)
                else:
                    for j in range(min(N_lst[i], dbar) + 1):
                        arr_rank1 = arr_rank[1:]
                        result = N_lst.copy()
                        result[i] -= j
                        #dbar -= j
                        get_before_reallocate(arr_rank1, num_dis, result, dbar-j, result_lst)
        
        def IBP(s):
            if s[-3]>0: return 1
            else: return 0
        def IDP(dp):
            if dp >0: return 1
            else: return 0
        def INi(ni):
            if ni > 0: return 1
            else: return 0
        def IS(s):
            if tuple(s) in State.keys(): return 1
            else: return 0
            
        def arrRateIn(s):
            n, col, val = 0, [], []
            for i in range(self.A):
                for j in range(self.A):
                    tempS = list(s)
                    # Ni, Rij
                    y = self.A + i * self.A + j
                    a1 = tempS[i] = tempS[i] + 1
                    a2 = tempS[y] = tempS[y] - 1
                    if IS(tempS):
                        n += 1
                        self.assertion(tempS)
                        col.append(State[tuple(tempS)])
                        val.append(self.ArrLst[i] * self.Pij[i][j])
            #         print(tempS)
            # print(n, col, val)
            # print('arr-----------')
            return n, col, val

        def backRateIn(s):
            n, col, val = 0, [], []
            for i in range(self.A):
                for j in range(self.A):
                    tempS = list(s)
                    # Ni, Rij
                    y = self.A + j * self.A + i
                    a1 = tempS[i] = tempS[i] - 1
                    a2 = tempS[y] = tempS[y] + 1
                    if IS(tempS):
                        n += 1
                        self.assertion(tempS)
                        col.append(State[tuple(tempS)])
                        val.append(self.RhoMtx[j][i] * a2 * (1 - self.Beta))
            #         print(tempS)
            # print(n, col, val)
            # print('back-----------')
            return n, col, val

        def broRateIn(s):
            n, col, val = 0, [], []
            for i in range(self.A):
                for j in range(self.A):
                    tempS = list(s)
                    # Rij, BP
                    y = self.A + j * self.A + i
                    a1 = tempS[y] = tempS[y] + 1
                    a2 = tempS[-3] = tempS[-3] - 1
                    if IS(tempS):
                        n += 1
                        self.assertion(tempS)
                        col.append(State[tuple(tempS)])
                        val.append(self.RhoMtx[j][i] * a1 * self.Beta)
            #         print(tempS)
            # print(n, col, val)
            # print('bro-----------')
            return n, col, val

        def gathRateIn(s):
            # 将坏车运送至维修中心
            n, col, val = 0, [], []
            for i in range(self.C+1):
                tempS = list(s)
                if i<self.C and tempS[-3]>0: continue
                tempS[-3] += i
                tempS[-2] -= i
                if tempS[-3] == 0: continue
                if IS(tempS):
                    n += 1
                    self.assertion(tempS)
                    col.append(State[tuple(tempS)])
                    val.append(self.Tau)
            #else: print('fucking wrong')
            return n, col, val

        def fixRateIn(s):
            n, col, val = 0, [], []
            tempS = list(s)
            # RC, DP
            a1 = tempS[-2] = tempS[-2] + 1
            a2 = tempS[-1] = tempS[-1] - 1
            if IS(tempS):
                n += 1
                self.assertion(tempS)
                col.append(State[tuple(tempS)])
                val.append(self.Mu * min(a1, self.N))
            # print(tempS)
            # print(n, col, val)
            # print('fix-----------')
            return n, col, val

        def redRateIn(s):
            n, col, val = 0, [], []
            for i in range(self.C+1):
                tempS = list(s)
                if i<self.C and tempS[-1]>0: continue
                tempS[-1] += i
                if tempS[-1] == 0: continue # 如果待投放量为0，则不可能发生投放
                result_lst, tail = [], tempS[self.A:]
                arr_rank, num_dis, N_lst1, dbar1 = np.argsort(self.ArrLst)[::-1], self.num_dis, tempS[:A], i
                get_before_reallocate(arr_rank, num_dis, N_lst1, dbar1, result_lst)
                for r in result_lst:
                    temp = r + tail
                    if IS(temp):
                        n += 1
                        self.assertion(temp)
                        col.append(State[tuple(temp)])
                        val.append(self.Tau)
            return n, col, val

        def getRateOut(s):
            outRate = 0
            for i in range(self.A):
                if INi(s[i]):
                    outRate += self.ArrLst[i]
                else:
                    continue
            for i in range(self.A):
                for j in range(self.A):
                    outRate += self.RhoMtx[i][j] * s[self.A + i * self.A + j]
            outRate += self.Mu * min(s[-2], self.N) + IBP(s)*self.Tau+IDP(s[-1])*self.Tau
            # inRate += sum(list(map(lambda a: a[0]*a[1], zip(x,y))))
            if outRate == 0: print(s, 'fucking wrong')
            return -outRate

        def getRateIn(s):
            n, col, val = 0, [], []
            '''
            # customer arrival: arrRateIn
            # ride back: backRateIn
            # ride break down: broRateIn
            # gathering: gathRateIn
            # fixing: fixRateIn
            # redistributing: redRateIn
            '''
            for f in [arrRateIn, backRateIn, broRateIn, gathRateIn, fixRateIn, redRateIn]:
                tempN, tempCol, tempVal = f(s)
                n += tempN
                col += tempCol
                val += tempVal
            if not val or 0 in val: print(s, 'fucking empty')
            return n, col, val

        def generateR():
            # R = csr_matrix((S,S), dtype=np.float)
            Row, Col, Value = [], [], []
            for k, s in enumerate(State):
                '''
                number of row: n
                row number: k
                column number: col
                value: data
                '''
                # 加1
                if k == n_state - 1:  # collect the last row as a test instance
                    tempN, tempCol, tempVal = getRateIn(s)
                    #if not tempVal: print(s)
                    tempCol += [k]
                    out = [getRateOut(s)]
                    #if not out: print(s)
                    tempVal += out
                else:  # generate the mtx
                    # set rate out for state s
                    Row += [k]
                    Col += [k]
                    out = [getRateOut(s)]
                    Value += out
                    #if not out: print(s)

                    # set rate in for state s
                    tempN, tempCol, tempVal = getRateIn(s)
                    Row += [k] * tempN
                    Col += tempCol
                    Value += tempVal
                    #if not tempVal: print(s)

            Row += [k] * n_state
            Col += list(range(n_state))
            Value += [1] * n_state
            R = csr_matrix((Value, (Row, Col)), dtype=np.float)  # .toarray()
            testArr = csr_matrix((tempVal, ([0] * (tempN + 1), tempCol)), dtype=np.float)
            return R, testArr

        return generateR()

    def getPortionState(self):
        BalanceMtx, testArr = self.getMtx()
        # print('getMtx done!')
        b = np.array([0] * (self.n_state - 1) + [1])
        x = spsolve(BalanceMtx, b)
        # %time x = dsolve.spsolve(BalanceMtx, b, use_umfpack=True)
        # %time x = dsolve.spsolve(BalanceMtx, b, use_umfpack=False)
        # x = isolve.cg(A=BalanceMtx, b=b)
        # assert(BalanceMtx.toarray()[-1].dot(x) == 1.0 and testArr.toarray().dot(x) == 0)
        portionState = {}
        for k, s in enumerate(self.State):
            assert x[k]<=1 and x[k]>=0
            portionState[s] = x[k]
        return portionState

    def getPerf(self):
        def getLost(state):
            _sum = 0.0
            for n, s in enumerate(range(self.A)):
                if state[n] == 0:
                    _sum += self.ArrLst[n]
            return _sum / sum(self.ArrLst)

        #         def getResult(n,b,l,i,B,R,D):
        #             re = []
        #             for _ in [n,b,l,i,B,R,D]:
        #                 re.append([self.para, _, 0])
        #             return re
        def getResult(l):
            return pd.DataFrame([l], columns=['para', 'n', 'b', 'l', 'i', 'B', 'R', 'D'])

        #self.para = para
        portionState = self.getPortionState()

        def convert_dict2data_frame(dictn):
            df = pd.DataFrame(list(dictn.keys()), columns=['a' + str(i) for i in range(self.A)] \
                                                          + ['r' + str(i) + str(j) for i in range(self.A) for j in
                                                             range(self.A)] + ['BP', 'RC', 'DP',])
            df['portion'] = dictn.values()
            return df

        #portionStateDf = convert_dict2data_frame(portionState)
        normalBikes, brokenBikes, idle, BP, RC, DP, lost = 0, 0, 0, 0, 0, 0, 0
        for k, s in enumerate(portionState):
            por = portionState[s]
            normalBikes += sum(s[:-3]) * por
            brokenBikes += sum(s[-3:]) * por
            if s[-2] == 0: idle += por
            BP += s[-3] * por
            RC += s[-2] * por
            DP += s[-1] * por
            lost += getLost(s) * por

        return normalBikes/self.M, lost, idle




def run(p):
    #print('run')
    result = []

    REPEAT = 5

    for epi in tqdm(range(REPEAT)):
        #np.random.seed(epi)
        #random.seed(epi)
        #def getPij(a):
        #    temp = np.random.rand(A,A)
        #    return (temp/sum(temp)).T
        #Pij = getPij(A)
        #ArrLst = np.random.rand(A)
        #print(Pij, ArrLst)
        for smt in np.arange(STEPSIZE*p+1, STEPSIZE*(p+1)+1, 1):
        #for i in range(1):
            #smt = 0.2
            #smt = round(smt, 1)
            para = N = smt
            instance = Model(A, M, Pij, ArrLst, RhoMtx, Beta, Tau, C, Mu, N)
            return_perf = list(instance.getPerf())
            return_perf += [epi, smt, para]
            result.append(return_perf)
            del instance, return_perf

    return result
#print(run(0))

def writeResult(re):
    df = pd.DataFrame(re, columns=['n','l','i','epi','smt','para'])
    df.to_csv(FileAdd+'NOde.csv')
    
# Hyper parameters
EPISODES = 30 # times every setting is replicated
# random.seed(1)
WriteFile = False
#WriteFile = True
NAME = 'Beta'
WARMTIME = 0
#RUNTIME = 80000
CORE = 4
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
    

    def send_notice(event_name, key, text):
        url = "https://maker.ifttt.com/trigger/"+event_name+"/with/key/"+key
        response = requests.request("POST", url)

        print(response.text)

    event_name = 'python_finished_running'
    key = 'bJK-bEYNYyBPXFnvNZ4P1x'
    text = ''
    send_notice(event_name, key, text)
