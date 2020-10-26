import csv
from math import factorial, isclose

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

# from disTest import A, ArrLst, Beta, M, Mu, N, Pij, RhoMtx, Theta

# generate state dictionary direrctly
def getStateDicts(A, M, Pij, ArrLst, RhoMtx, Beta, Mu, N, Theta):
    def temp(x, i, index, l):
        x.append(i)
        generate_R(x.copy(), index, l)
        
    def generate_R(s, index, l):
        if len(s)==A+A+A**2 and sum(s)==M:
            State[tuple(s+[l])] = index[0]
            index[0] += 1
        elif len(s)>A+3+A**2 or sum(s)>M:
            return 0
        else:
            for i in range(M+1):
                temp(s.copy(), i, index, l)

    def getState():            
        index = [0]
        for l in range(A):
            for i in range(M+1):
                generate_R([i], index, l)
        return State


    # 使用scipy.sparse.csr_sparse
    # generate R matrix

    def INi(ni):
        if ni > 0: return 1
        else: return 0
    def IL(s):
        if s[A+s[-1]] == 0: return 1
        else: return 0
    def IS(l):
        # 判断是不是都是在0到M之间
        if len(l) == sum(list(map(lambda x: M >= x >= 0, l))): return 1
        else: return 0

    def arrRateIn(s):
        n, col, val = 0, [], []
        for i in range(A):
            for j in range(A):
                tempS = list(s)
                # Ni, Rij
                y = 2*A + i*A + j
                a1 = tempS[i] = tempS[i] + 1
                a2 = tempS[y] = tempS[y] - 1
                if IS([a1, a2]):
                    n += 1
                    col.append(State[tuple(tempS)])
                    val.append(ArrLst[i]*Pij[i][j])
        #         print(tempS)
        # print(n, col, val)
        # print('arr-----------')
        return n, col, val
                
    def backRateIn(s):
        n, col, val = 0, [], []
        for i in range(A):
            for j in range(A):
                tempS = list(s)
                # Ni, Rij
                y = 2*A + j*A + i
                a1 = tempS[i] = tempS[i] - 1
                a2 = tempS[y] = tempS[y] + 1
                if IS([a1, a2]):
                    n += 1
                    col.append(State[tuple(tempS)])
                    val.append(RhoMtx[j][i]*a2*(1-Beta))
        #         print(tempS)
        # print(n, col, val)
        # print('back-----------')
        return n, col, val

    def broRateIn(s):
        n, col, val = 0, [], []
        for i in range(A):
            for j in range(A):
                tempS = list(s)
                #Rij, BP
                y = 2*A + j*A + i
                a1 = tempS[y] = tempS[y] + 1
                a2 = tempS[A+i] = tempS[A+i] - 1
                if IS([a1, a2]):
                    n += 1
                    col.append(State[tuple(tempS)])
                    val.append(RhoMtx[j][i]*a1*Beta)
        #         print(tempS)
        # print(n, col, val)
        # print('bro-----------')
        return n, col, val

    def fixRateIn(s):
        n, col, val = 0, [], []
        tempS = list(s)
        # RC, DP
        L = s[-1]
        a1 = tempS[A+L] = tempS[A+L] + 1
        a2 = tempS[L] = tempS[L] - 1
        if IS([a1, a2]):
            n += 1
            col.append(State[tuple(tempS)])
            val.append(Mu*min(tempS[A+L], N))
        # print(tempS)
        # print(n, col, val)
        # print('fix-----------')
        return n, col, val

    def movRateIn(s):
        n, col, val = 0, [], []
        tempS = list(s)
        index = (s[-1]-1) % A
        tempS[A+index] = 0
        tempS[-1] = index
        if sum(tempS[:-1]) == M:
            n += 1
            col.append(State[tuple(tempS)])
            val.append(Theta)
        # print(tempS)
        # print(n, col, val)
        # print('red-----------')
        return n, col, val
        
    def getRateOut(s):
        outRate = 0
        for i in range(A):
            if INi(s[i]):
                outRate += ArrLst[i]
            else: continue
        for i in range(A):
            for j in range(A):
                outRate += RhoMtx[i][j] * s[2*A+i*A+j]
        outRate += Mu*min(s[A+s[-1]], N) + Theta*IL(s)
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
        for f in [arrRateIn, backRateIn, broRateIn, fixRateIn, movRateIn]:
            tempN, tempCol, tempVal = f(s)
            n += tempN
            col += tempCol
            val += tempVal
        
        return n, col, val

    def generateRMtx():
        #R = csr_matrix((S,S), dtype=np.float)
        Row, Col, Value = [], [], []
        for k, s in enumerate(State):
            '''
            number of row: n
            row number: k
            column number: col
            value: data
            '''
            # 加1
            if k==n_state-1: # collect the last row as a test instance
                tempN, tempCol, tempVal = getRateIn(s)
                tempCol += [k]
                tempVal += [getRateOut(s)]
            else:            # generate the mtx
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
        R = csr_matrix((Value, (Row, Col)), dtype=np.float) #.toarray()
        testArr = csr_matrix((tempVal, ([0]*(tempN+1), tempCol)), dtype=np.float)
        return R, testArr


    State = {}
    getState()
    n_state = len(State)
    BalanceMtx, testArr = generateRMtx()
    b = np.array([0]*(n_state-1) + [1])
    x = spsolve(BalanceMtx, b)
    portionState = {}
    for k,s in enumerate(State):
        portionState[s] = x[k]
    return State, portionState

#print(portionState)
