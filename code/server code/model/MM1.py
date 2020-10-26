import numpy as np
import pandas as pd

T = 0
N = 100000
np.random.seed(1)
a = np.cumsum(np.random.exponential(0.2, N))
#a
s = np.random.exponential(0.1, N)
#s

inservice = []
rw = pd.DataFrame(index=range(N), columns = ['arrival', 'serve', 'leave'])
rl, l = pd.DataFrame(columns=['n_system', 't_system']), 0
rq, q = pd.DataFrame(columns=['n_queue', 't_queue']), 0

for n, t in enumerate(a):
    if inservice:
        if t<=inservice[0][0]:
            T = t
            rl.loc[l], l = [len(inservice), T], l+1
            rq.loc[q], q = [len(inservice)-1, T], q+1
            rw.iloc[n] = [T, inservice[-1][0], inservice[-1][0]+s[n]]
            inservice.append([inservice[-1][0]+s[n], n])
            inservice.sort()
        else:
            while inservice[0][0] < t:
                T = inservice[0][0]
                rl.loc[l], l = [len(inservice), T], l+1
                rq.loc[q], q = [len(inservice)-1, T], q+1
                inservice.pop(0)
                if not inservice:
                    rq.loc[q], q = [0, t], q+1
                    break
            T = t
            rl.loc[l], l = [len(inservice), T], l+1
            if inservice:
                rw.iloc[n] = [T, inservice[-1][0], inservice[-1][0]+s[n]]
                inservice.append([inservice[-1][0]+s[n], n])
            else:
                rw.iloc[n] = [T, T, T+s[n]]
                inservice.append([T+s[n], n])
            inservice.sort()
    else:
        T = t
        rw.iloc[n] = [T, T, T+s[n]]
        rl.loc[l], l = [0, T], l+1
        inservice.append([T+s[n], n])
        inservice.sort()
rw.to_csv('C:/Rebalancing/data/result/tandemPropaDelay/rw.csv')
rl1.to_csv('C:/Rebalancing/data/result/tandemPropaDelay/rl1.csv')
rl2.to_csv('C:/Rebalancing/data/result/tandemPropaDelay/rl2.csv')
rq1.to_csv('C:/Rebalancing/data/result/tandemPropaDelay/rq1.csv')
rq2.to_csv('C:/Rebalancing/data/result/tandemPropaDelay/rq2.csv')