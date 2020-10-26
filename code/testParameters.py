# cen parameters
import numpy as np
import random as random

# set parameters
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


Instance = 4

np.random.seed(Instance)
random.seed(0)

A = 10
M = 50
FileAdd = 'C:\\Rebalancing\\nowModel\\result\\A'+str(A)+'M'+str(M)

def getPij(a):
    temp = np.log1p(np.random.rand(A,A))
    return (temp/sum(temp)).T
Pij = getPij(A)
ArrLst = np.log1p((np.random.rand(A)))
Beta = 0.3
#RhoMtx = [[1.0] * A for i in range(A)]
#RhoMtx = np.random.rand(A, A)

Gamma = 1
Mu = 1
N = 1
Delta = 1
B_, D_ = 2, 2

# RhoMtx = [[1.0, 0.5], 
#           [0.5, 1.0]]
RhoMtx = [[1/(abs(j-i)+1) for i in range(A)] for j in range(A)]