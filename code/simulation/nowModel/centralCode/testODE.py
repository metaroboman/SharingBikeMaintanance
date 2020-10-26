import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from math import factorial
from math import isclose
from tqdm import tqdm
import csv
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from testParameters import A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_

def getStateDicts(A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_):
    #print(A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_)
    def temp(x, i, index, possSet):
        x.append(i)
        generate_R(x.copy(), index, possSet)

    def generate_R(s, index, possSet):
        L = A+3+A**2
        if len(s)==L and sum(s)==M and sum(s[-2:]) in possSet:
            State[tuple(s)] = index[0]
            index[0] += 1
        elif len(s)>L or sum(s)>M or (len(s)==L and not sum(s[-2:]) in possSet):
            return 0
        else:
            for i in range(M+1):
                temp(s.copy(), i, index, possSet)
                
    def getPossSet():
        possSet = []
        inLst = [i for i in range(0, M+1, B_)]
        #print(inLst)
        for _ in inLst:
            fake = [i for i in range(_, -1, -D_)]
            possSet += fake
            #print(fake)
        return set(possSet)

    def getState():
        # generate state dictionary direrctly
        State = {}
        index = [0]
        possSet = getPossSet()
        for i in range(M+1):
            generate_R([i], index, possSet)
        return State

    def INi(ni):
        if ni > 0: return 1
        else: return 0
    def IBP(bp):
        if bp >= B_: return 1
        else: return 0
    def IDP(dp):
        if dp >= D_: return 1
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
                y = A + i*A + j
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
                y = A + j*A + i
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
                y = A + i*A + j
                a1 = tempS[y] = tempS[y] + 1
                a2 = tempS[-3] = tempS[-3] - 1
                if IS([a1, a2]):
                    n += 1
                    col.append(State[tuple(tempS)])
                    val.append(RhoMtx[i][j]*a1*Beta)
        #         print(tempS)
        # print(n, col, val)
        # print('bro-----------')
        return n, col, val

    def gathRateIn(s):
        n, col, val = 0, [], []
        tempS = list(s)
        # BP, RC
        a1 = tempS[-3] = tempS[-3] + B_
        a2 = tempS[-2] = tempS[-2] - B_
        if IS([a1, a2]):
            n += 1
            col.append(State[tuple(tempS)])
            val.append(Gamma)
        # print(tempS)
        # print(n, col, val)
        # print('gath-----------')
        return n, col, val

    def fixRateIn(s):
        n, col, val = 0, [], []
        tempS = list(s)
        # RC, DP
        a1 = tempS[-2] = tempS[-2] + 1
        a2 = tempS[-1] = tempS[-1] - 1
        if IS([a1, a2]):
            n += 1
            col.append(State[tuple(tempS)])
            val.append(Mu*min(a1, N))
        # print(tempS)
        # print(n, col, val)
        # print('fix-----------')
        return n, col, val

    def redRateIn(s):
        n, col, val = 0, [], []
        tempS = list(s)
        # DP, Ni
        for i in range(A):
            tempS[i] = tempS[i] - D_/A
        tempS[-1] += D_
        if IS(tempS):
            n += 1
            col.append(State[tuple(tempS)])
            val.append(Delta)
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
                outRate += RhoMtx[i][j] * s[A+i*A+j]
        outRate += Gamma*IBP(s[-3]) + Mu*min(s[-2], N) + Delta*IDP(s[-1])
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
#portionState = getStateDicts(A, M, Pij, ArrLst, RhoMtx, Beta, B_, Gamma, Mu, N, Delta, D_)[1]