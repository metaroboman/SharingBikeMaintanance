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

A = 2
M = 6
Pij = [[0.3, 0.7],
       [0.7, 0.3]]
# Pij = [[0.1, 0.2, 0.3, 0.4],
#        [0.2, 0.3, 0.4, 0.1],
#        [0.3, 0.4, 0.1, 0.2],
#        [0.4, 0.3, 0.2, 0.1]]
Beta = 0.3
ArrLst = [5.0, 5.0]
#ArrLst = [5.0, 5.0, 6.0, 7.0]
Gamma = 1.0
Mu = 1.0
Delta = 1.0
# RhoMtx = [[1.0, 1.0], 
#           [1.0, 1.0]]
RhoMtx = [[1.0] * A for i in range(A)]
N = 1
B_, D_ = 2, 2