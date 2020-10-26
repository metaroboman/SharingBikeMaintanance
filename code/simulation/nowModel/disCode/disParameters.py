# set the parameters
# get the number of states S
'''
A: number of areas
M: number of total bikes
S: number of total states
Pij: transfering possibility matrix
Beta: broken rate
ArrMtx: arrival rates of each area
Theta: moving rate
Mu: fix rate
RhoMtx: matrix of ride rates
N: number of fix servers
'''

A = 2
M = 6

Pij = [[0.3, 0.7],
       [0.7, 0.3]]
Beta = 0.3
ArrLst = [5.0, 5.0]
Theta = 1.0
Mu = 1.0
RhoMtx = [[1.0, 1.0], 
          [1.0, 1.0]]
N =1