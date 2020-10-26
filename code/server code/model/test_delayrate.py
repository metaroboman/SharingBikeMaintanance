# closed network with two queues and propagation delay
import numpy as np
import pandas as pd

configuration = []
for i in range(5,15):
    for j in range(5, 15):
        configuration.append([i*0.1, j*0.1])

result, R = pd.DataFrame(columns=['miu1', 'miu2', 'de', 'arr_avg_1', 'arr_avg_2']), 0
for setting in configuration:
    Left = min(setting[0], setting[1]) * 0.9
    Right = 1.1 * max(setting[0], setting[1])
    for de in np.arange(Left, Right, (Right - Left) / 10):
        T = 0
        N = 15000
        n_customer = 20
        np.random.seed(1)
        p_delay = np.random.exponential(de, N)
        miu1 = setting[0]
        miu2 = setting[1]



        _ = np.random.exponential(miu1, N)
        ins1 = [sum(_[:i + 1]) for i in range(n_customer)]
        ins2 = []
        delay = []
        scheduler = [[s, 1] for s in ins1]

        rw, W = pd.DataFrame(columns=['position', 'arr', 'ser', 'lea']), 5
        rl1, L1 = pd.DataFrame(columns=['n_sys', 't_sys']), 0
        rld, Ld = pd.DataFrame(columns=['n_sys', 't_sys']), 0
        rl2, L2 = pd.DataFrame(columns=['n_sys', 't_sys']), 0
        for i in range(5):
            rw.loc[i] = [1, 0, sum(_[:i]), sum(_[:i]) + _[i]]
        for n in range(N):

            prop_delay = p_delay[n]
            T, obj = scheduler[0]

            if obj == 1:
                rl1.loc[L1], L1 = [len(ins1), T], L1 + 1
                rld.loc[Ld], Ld = [len(delay), T], Ld + 1
                rw.loc[W], W = [0, T, T, T + prop_delay], W + 1
                delay.append(T + prop_delay)
                scheduler.append([T + prop_delay, 0])
                ins1.pop(0)
            elif obj == 2:
                rl2.loc[L2], L2 = [len(ins2), T], L2 + 1
                rl1.loc[L1], L1 = [len(ins1), T], L1 + 1
                if ins1:
                    rw.loc[W], W = [1, T, ins1[-1], ins1[-1] + np.random.exponential(miu1)], W + 1
                else:
                    rw.loc[W], W = [1, T, T, T + np.random.exponential(miu1)], W + 1
                ins1.append(rw.iloc[W - 1][3])
                scheduler.append([rw.iloc[W - 1][3], 1])
                ins2.pop(0)
            else:
                rld.loc[Ld], Ld = [len(delay), T], Ld + 1
                rl2.loc[L2], L2 = [len(ins2), T], L2 + 1
                if ins2:
                    rw.loc[W], W = [2, T, ins2[-1], ins2[-1] + np.random.exponential(miu2)], W + 1
                else:
                    rw.loc[W], W = [2, T, T, T + np.random.exponential(miu2)], W + 1
                ins2.append(rw.iloc[W - 1][3])
                scheduler.append([rw.iloc[W - 1][3], 2])
                delay.pop(0)
            scheduler.pop(0)
            scheduler.sort()

        result.loc[R], R = [setting[0], setting[1], de, np.average(rw[rw.position == 1.0].arr.diff()[1:]),
                        np.average(rw[rw.position == 2.0].arr.diff()[1:])], R + 1
    result.to_csv('C:/Rebalancing/data/result/tandemPropaDelay1/testresult.csv')
    print('over')