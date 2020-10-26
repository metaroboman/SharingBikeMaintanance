#2 M/M/1 open tandem queueing system with determined propagation delay
import numpy as np
import pandas as pd

T = 0
N = 20000
np.random.seed(1)
prop_delay = 0
a = np.cumsum(np.random.exponential(0.2, N))
s1 = np.random.exponential(1/10, N)
s2 = np.random.exponential(1/10, N)


ins1 = []
ins2 = []
rw = pd.DataFrame(index=range(N), columns = ['arrl', 'ser1', 'lea1', 'arr2', 'ser2', 'lea2'])
rl1, L1 = pd.DataFrame(columns=['n_sys', 't_sys']), 0
rq1, Q1 = pd.DataFrame(columns=['n_que', 't_que']), 0
rl2, L2 = pd.DataFrame(columns=['n_sys', 't_sys']), 0
rq2, Q2 = pd.DataFrame(columns=['n_que', 't_que']), 0

for n, t in enumerate(a):
    T = t
    if ins1:
        while ins1[0] <= T:
            rl1.loc[L1], L1 = [len(ins1), ins1[0]], L1+1
            rq1.loc[Q1], Q1 = [len(ins1)-1, ins1[0]], Q1+1
            ins1.pop(0)
            if not ins1: break
    if ins2:
        while ins2[0][0] <= T:
            L = sum(pd.DataFrame(ins2).loc[:,1]<=T)
            rl2.loc[L2], L2 = [L, ins2[0][0]], L2+1
            rq2.loc[Q2], Q2 = [L-1, ins2[0][0]], Q2+1
            ins2.pop(0)
            if not ins2: break

            
    if ins1:
        if ins2:
            rw.iloc[n] = [T, ins1[-1], ins1[-1]+s1[n], ins1[-1]+s1[n]+prop_delay, ins2[-1][0], ins2[-1][0]+s2[n]]
            while ins2[0][0] <= ins1[-1]+s1[n]+prop_delay:
                L = sum(pd.DataFrame(ins2).loc[:,1]<=ins1[-1]+s1[n]+prop_delay)
                rl2.loc[L2], L2 = [L, ins2[0][0]], L2+1
                rq2.loc[Q2], Q2 = [L-1, ins2[0][0]], Q2+1
                ins2.pop(0)
                if not ins2: break
            rl2.loc[L2], L2 = [len(ins2), ins1[-1]+s1[n]+prop_delay], L2+1
            rq2.loc[Q2], Q2 = [len(ins2)-1, ins1[-1]+s1[n]+prop_delay], Q2+1 
            if ins2: ins2.append([ins2[-1][0]+s2[n], ins1[-1]+s1[n]+prop_delay])
            else: ins2.append([ins1[-1]+s1[n]+prop_delay+s2[n], ins1[-1]+s1[n]+prop_delay])
            ins2.sort()
        else: 
            rw.iloc[n] = [T, ins1[-1], ins1[-1]+s1[n], ins1[-1]+s1[n]+prop_delay, ins1[-1]+s1[n]+prop_delay, ins1[-1]+s1[n]+prop_delay+s2[n]]
            rl2.loc[L2], L2 = [0, ins1[-1]+s1[n]+prop_delay], L2+1
            ins2.append([T+s1[n]+prop_delay+s2[n], T+s1[n]+prop_delay])
            ins2.sort()
            
        rl1.loc[L1], L1 = [len(ins1), T], L1+1
        rq1.loc[Q1], Q1 = [len(ins1)-1, T], Q1+1    
        ins1.append(ins1[-1]+s1[n])
        ins1.sort()
        
    else:
        if ins2:
            rw.iloc[n] = [T, T, T+s1[n], T+s1[n]+prop_delay, ins2[-1][0], ins2[-1][0]+s2[n]]
            while ins2[0][0] <= T+s1[n]+prop_delay:
                L = sum(pd.DataFrame(ins2).loc[:,1]<=T+s1[n]+prop_delay)
                rl2.loc[L2], L2 = [L, ins2[0][0]], L2+1
                rq2.loc[Q2], Q2 = [L-1, ins2[0][0]], Q2+1
                ins2.pop(0)
                if not ins2: break
            rl2.loc[L2], L2 = [len(ins2), T+s1[n]+prop_delay], L2+1
            rq2.loc[Q2], Q2 = [len(ins2)-1, T+s1[n]+prop_delay], Q2+1 
            if ins2: ins2.append([ins2[-1][0]+s2[n], T+s1[n]+prop_delay])
            else: ins2.append([T+s1[n]+prop_delay+s2[n], T+s1[n]+prop_delay])
            ins2.sort()
        else: 
            rw.iloc[n] = [T, T, T+s1[n], T+s1[n]+prop_delay, T+s1[n]+prop_delay, T+s1[n]+prop_delay+s2[n]]
            rl2.loc[L2], L2 = [0, T+s1[n]+prop_delay], L2+1
            ins2.append([T+s1[n]+prop_delay+s2[n], T+s1[n]+prop_delay])
            ins2.sort()
    
        rl1.loc[L1], L1 = [0, T], L1+1
        ins1.append(T+s1[n])
        ins1.sort()
        
rw.to_csv('C:/Rebalancing/data/result/tandemPropaDelay/rw.csv')
rl1.to_csv('C:/Rebalancing/data/result/tandemPropaDelay/rl1.csv')
rl2.to_csv('C:/Rebalancing/data/result/tandemPropaDelay/rl2.csv')
rq1.to_csv('C:/Rebalancing/data/result/tandemPropaDelay/rq1.csv')
rq2.to_csv('C:/Rebalancing/data/result/tandemPropaDelay/rq2.csv')