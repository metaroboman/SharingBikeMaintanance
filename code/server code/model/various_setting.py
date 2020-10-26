# closed network with two queues and propagation delay
import numpy as np
import pandas as pd
import seaborn as sb

configuration = [[0.5, 0, 0.5],
                 [0.5, 0, 1.5],
                 [1.5, 0, 0.5],
                 [0.5, 1, 0.5],
                 [0.5, 1, 1.5],
                 [1.5, 1, 0.5]]

for setting in configuration:
    for de in [(setting[0]+setting[1])/4, (setting[0]+setting[1])/2, (setting[0]+setting[1])*0.75]:
        T = 0
        N = 20000
        n_customer = 10
        np.random.seed(1)
        if setting[1] == 0:
            p_delay = [de] * N
        else:
            p_delay = np.random.exponential(de, N)
        miu1 = setting[0]
        miu2 = setting[2]

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
    
        rw.to_csv('C:/Rebalancing/data/result/tandemPropaDelay1/'+str(setting[0])+'_'+str(setting[1])+'_'+str(setting[2])+'_'+str(de)+'_'+'rw.csv')
        rl1.to_csv('C:/Rebalancing/data/result/tandemPropaDelay1/'+str(setting[0])+'_'+str(setting[1])+'_'+str(setting[2])+'_'+str(de)+'_'+'rl1.csv')
        rld.to_csv('C:/Rebalancing/data/result/tandemPropaDelay1/'+str(setting[0])+'_'+str(setting[1])+'_'+str(setting[2])+'_'+str(de)+'_'+'rld.csv')
        rl2.to_csv('C:/Rebalancing/data/result/tandemPropaDelay1/'+str(setting[0])+'_'+str(setting[1])+'_'+str(setting[2])+'_'+str(de)+'_'+'rl2.csv')
    
        print('over')