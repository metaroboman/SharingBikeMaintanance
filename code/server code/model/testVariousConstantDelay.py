# closed network with two queues and 2 propagation delay processes

import numpy as np
import pandas as pd
from tqdm import tqdm

configuration = [[1.0, 1.0],
                 [2.0, 2.0],
                 [3.0, 3.0],
                 [1.0, 2.0],
                 [1.0, 3.0]]

re, index = pd.DataFrame(columns=['miu1', 'miu2', 'd1', 'd2', 'arr1', 'arr2', 'arrd1', 'arrd2', 'p10', 'p20', 'pd10',
                                 'pd20']), 0

for setting in tqdm(configuration):
    L, R = min(setting), max(setting)
    for de in [0.0, 1.0/L, 2.0/(L+R), 1.0/R, 10.0/R, 100.0/R]:
        for i in [0,1]:
            T = 0
            N = 40000
            n_customer = 100
            np.random.seed(1)
            if i:
                prop_delay1 = prop_delay2 = de
            else:
                prop_delay1 = 0.5 * de
                prop_delay2 = 1.5 * de
            miu1 = setting[0]
            miu2 = setting[1]

            _ = np.random.exponential(miu1, N)
            ins1 = [sum(_[:i+1]) for i in range(n_customer)]
            ins2 = []
            delay1 = []
            delay2 = []
            scheduler = [[s, 1] for s in ins1]

            rw, W = pd.DataFrame(columns = ['place', 'arr', 'ser', 'lea']), n_customer
            rl1, L1 = pd.DataFrame(columns=['n_sys', 't_sys']), 0
            rld1, Ld1 = pd.DataFrame(columns=['n_sys', 't_sys']), 0
            rl2, L2 = pd.DataFrame(columns=['n_sys', 't_sys']), 0
            rld2, Ld2 = pd.DataFrame(columns=['n_sys', 't_sys']), 0

            for i in range(n_customer):
                rw.loc[i] = [1, 0, sum(_[:i]), sum(_[:i])+_[i]]

            for n in range(N):

                T, obj = scheduler[0]

                if obj == 1:
                    rl1.loc[L1], L1 = [len(ins1), T], L1+1
                    rld1.loc[Ld1], Ld1 = [len(delay1), T], Ld1+1
                    rw.loc[W], W = [-1, T, T, T+prop_delay1], W+1
                    delay1.append(T+prop_delay1)
                    scheduler.append([T+prop_delay1, -1])
                    ins1.pop(0)
                elif obj == 2:
                    rl2.loc[L2], L2 = [len(ins2), T], L2+1
                    rld2.loc[Ld2], Ld2 = [len(delay2), T], Ld2+1
                    rw.loc[W], W = [-2, T, T, T+prop_delay2], W+1
                    delay2.append(T+prop_delay2)
                    scheduler.append([T+prop_delay2, -2])
                    ins2.pop(0)
                elif obj == -1:
                    rld1.loc[Ld1], Ld1 = [len(delay1), T], Ld1+1
                    rl2.loc[L2], L2 = [len(ins2), T], L2+1
                    if ins2: rw.loc[W], W = [2, T, ins2[-1], ins2[-1]+np.random.exponential(miu2)], W+1
                    else: rw.loc[W], W = [2, T, T, T+np.random.exponential(miu2)], W+1
                    ins2.append(rw.iloc[W-1][3])
                    scheduler.append([rw.iloc[W-1][3], 2])
                    delay1.pop(0)
                else:
                    rld2.loc[Ld2], Ld2 = [len(delay2), T], Ld2+1
                    rl1.loc[L1], L1 = [len(ins1), T], L1+1
                    if ins1: rw.loc[W], W = [1, T, ins1[-1], ins1[-1]+np.random.exponential(miu1)], W+1
                    else: rw.loc[W], W = [1, T, T, T+np.random.exponential(miu1)], W+1
                    ins1.append(rw.iloc[W-1][3])
                    scheduler.append([rw.iloc[W-1][3], 1])
                    delay2.pop(0)
                scheduler.pop(0)
                scheduler.sort()
                
            arr1 = np.average(rw[rw.place == 1.0].arr)
            arr2 = np.average(rw[rw.place == 2.0].arr)
            arrd1 = np.average(rw[rw.place == -1.0].arr)
            arrd2 = np.average(rw[rw.place == -2.0].arr)
            p10 = sum(rl1.t_sys.diff()[rl1.n_sys == 0.0][1:])/T
            p20 = sum(rl2.t_sys.diff()[rl2.n_sys == 0.0][1:])/T
            pd10 = sum(rld1.t_sys.diff()[rld1.n_sys == 0.0][1:])/T
            pd20 = sum(rld2.t_sys.diff()[rld2.n_sys == 0.0][1:])/T
            re.loc[index], index = [miu1, miu2, prop_delay1, prop_delay2, arr1, arr2, arrd1, arrd2, p10, p20, pd10, pd20], index+1
            del T, N, n_customer, _, ins1, ins2, delay1, delay2, scheduler, rw, W, rl1, L1, rld1, Ld1, rl2, L2, rld2, Ld2
            del miu1, miu2, prop_delay1, prop_delay2, arr1, arr2, arrd1, arrd2, p10, p20, pd10, pd20

re.to_csv('constantDelayResult.csv')