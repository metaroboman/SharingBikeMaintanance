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
from time import time
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import random


class perfCenODE():
    def __init__(self, A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_):
        # init parameters
        self.A, self.M, self.Pij, self.ArrLst, self.RhoMtx, self.Beta, self.B_, self.Gamma, self.Mu, self.N, self.Delta, self.D_ = \
            A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_
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

    def getMtx(self):
        State = self.State
        n_state = self.n_state

        def get_target_number():
            A, M, D_, ArrLst = self.A, self.M, self.D_, self.ArrLst
            arr = np.array(ArrLst) / sum(ArrLst)
            num_dis = [int(M * x) for x in arr]
            left = M - sum(num_dis)
            for i in np.argsort(ArrLst)[:left]:
                num_dis[i] += 1
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

        #         A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_ = \
        #         self.A, self.M, self.Pij, self.ArrLst, self.RhoMtx, self.Beta, self.B_, self.Gamma, self.Mu, self.N, self.Delta, self.D_
        def INi(ni):
            if ni > 0:
                return 1
            else:
                return 0
        def IBP(s):
            if s[-3] >= self.B_ or (sum(s[-3:]) == self.M and s[-3] > 0):
                return 1
            else:
                return 0
        def IDP(s):
            if s[-1] >= self.D_ or (sum(s[-3:]) == self.M and s[-1] > 0):
                return 1
            else:
                return 0
        def IS(s):
            if tuple(s) in State.keys():
                return 1
            else:
                return 0

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
                    y = self.A + i * self.A + j
                    a1 = tempS[y] = tempS[y] + 1
                    a2 = tempS[-3] = tempS[-3] - 1
                    if IS(tempS):
                        n += 1
                        col.append(State[tuple(tempS)])
                        val.append(self.RhoMtx[i][j] * a1 * self.Beta)
            #         print(tempS)
            # print(n, col, val)
            # print('bro-----------')
            return n, col, val

        def gathRateIn(s):
            n, col, val = 0, [], []
            tempS = list(s)
            b = []
            # BP, RC
            if s[-2]>0: # only if there exists bikes being repaired
                if sum(s[-3:])!=self.M:
                    b += [self.B_]
                else:
                    for i in range(s[-2]):
                        b += [i+1]
                for j in b:
                    tempS[-3] += j
                    tempS[-2] -= j
                    if IS(tempS):
                        n += 1
                        col.append(State[tuple(tempS)])
                        val.append(self.Gamma)

            return n, col, val

        def fixRateIn(s):
            n, col, val = 0, [], []
            tempS = list(s)
            # RC, DP
            a1 = tempS[-2] = tempS[-2] + 1
            a2 = tempS[-1] = tempS[-1] - 1
            if IS(tempS):
                n += 1
                col.append(State[tuple(tempS)])
                val.append(self.Mu * min(a1, self.N))
            # print(tempS)
            # print(n, col, val)
            # print('fix-----------')
            return n, col, val

        def redRateIn(s):
            n, col, val = 0, [], []
            if sum(s[-3:]) == self.M: return n, col, val
            tempS = list(s)
            A, D_, num_dis = self.A, self.D_, self.num_dis
            # DP, Ni
            arr_rank, N_lst1, dbar1, result_lst = np.argsort(self.ArrLst)[::-1], tempS[:A], D_, []
            get_before_reallocate(arr_rank, num_dis, N_lst1, dbar1, result_lst)
            tempS[-1] += D_
            tail = tempS[A:]
            for x in result_lst:
                temp = x + tail
                if IS(temp):
                    n += 1
                    col.append(State[tuple(temp)])
                    val.append(Delta)
            # print(tempS)
            # print(n, col, val)
            # print('red-----------')
            del arr_rank, N_lst1, dbar1, result_lst, tail
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
            outRate += self.Gamma * IBP(s) + self.Mu * min(s[-2], self.N) + self.Delta * IDP(s)
            # inRate += sum(list(map(lambda a: a[0]*a[1], zip(x,y))))
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
                    tempCol += [k]
                    tempVal += [getRateOut(s)]
                else:  # generate the mtx
                    # set rate out for state s
                    Row += [k]
                    Col += [k]
                    Value += [getRateOut(s)]

                    # set rate in for state s
                    tempN, tempCol, tempVal = getRateIn(s)
                    Row += [k] * tempN
                    Col += tempCol
                    Value += tempVal

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
            portionState[s] = x[k]
        return portionState

    def getPerf(self, para):
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
            return pd.DataFrame([l], columns=['para', 'n', 'l', 'i', 'B', 'R', 'D'])

        self.para = para
        portionState = self.getPortionState()

        def convert_dict2data_frame(dictn):
            df = pd.DataFrame(list(dictn.keys()), columns=['a' + str(i) for i in range(self.A)] \
                                                          + ['r' + str(i) + str(j) for i in range(self.A) for j in
                                                             range(self.A)] + ['BP', 'RC', 'DP'])
            df['portion'] = dictn.values()
            return df

        portionStateDf = convert_dict2data_frame(portionState)
        normalBikes, brokenBikes, idle, BP, RC, DP, lost = 0, 0, 0, 0, 0, 0, 0
        for k, s in enumerate(portionState):
            por = portionState[s]
            normalBikes += sum(s[:-3]) * por
            # brokenBikes += sum(s[-3:]) * por
            if s[-2] == 0: idle += por
            BP += s[-3] * por
            RC += s[-2] * por
            DP += s[-1] * por
            lost += getLost(s) * por
        return portionStateDf, getResult(
            [para, normalBikes / self.M, lost, idle, BP / self.M, RC / self.M, DP / self.M])

if __name__ == '__main__':
    Instance = 4
    np.random.seed(Instance)
    random.seed(0)
    A = 2
    M = 8
    def getPij(a):
        temp = np.log1p(np.random.rand(A, A))
        return (temp / sum(temp)).T
    Pij = getPij(A)
    ArrLst = np.log1p(np.random.rand(A))
    Beta = 0.4
    Gamma = 1
    Mu = 1
    N = 1
    Delta = 1
    B_, D_ = 2, 2
    RhoMtx = [[1.0, 0.5],
              [0.5, 1.0]]
    # RhoMtx = [[1/(abs(j-i)+1) for i in range(A)] for j in range(A)]

    instance = perfCenODE(A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_)
    print(instance.getPerf(Mu)[1])