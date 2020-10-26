import numpy as np
import pandas as pd


class Area():
    def __init__(self, n, a_id):
        self.a_id = a_id
        self.normal_bike = n
        self.broken_bike = 0

    def move(self):
        self.normal_bike -= 1
        self.broken_bike += 1

    def repair(self):
        self.normal_bike += 1
        self.broken_bike -= 1


def binaryInsert(target, events):
    for event in events:
        if event >= target[-1]:
            target.append(event)
        else:
            l, mid, r = 0, int(len(target) / 2), len(target) - 1
            while 1:
                if r - l == 1:
                    target.insert(r, event)
                    break
                else:
                    if event > target[mid]:
                        l = mid
                        mid = int((r + l) / 2)
                    else:
                        r = mid
                        mid = int((r + l) / 2)


class BikeNet():
    def __init__(self, N, R, A, Q, repair, time_limit, start_position):
        self.N = N
        self.R = R
        self.A = A
        self.Q = Q
        self.repair = repair
        self.time_limit = time_limit
        self.carrier_position = start_position
        self.reset()
        self.trans = {}

    def reset(self):
        # self.__init__(self.N, self.R, self.A, self.Q, self.P, self.time_limit)
        # stat, S = pd.DataFrame(columns=['type', 'place', 't']), 0
        # loss, L = pd.DataFrame(columns=['place', 't']), 0
        # broken, B = pd.DataFrame(columns=['place', 'ng', 'nb', 't']), 0

        # initiation of instances of Area and scheduler
        self.T = 0
        self.scheduler = []
        self.a = []  # list of instances of areas
        self.s = np.array([(self.N / self.A) if i % 2 == 0 else 0 for i in range(2 * self.A)])
        for i in range(self.A):
            self.a.append(Area(self.N / self.A, i))
            self.scheduler.append([np.random.exponential(1 / self.R.loc[i].cus_arr), 1, self.a[i]])
        self.scheduler.sort()

        return self.s.copy()
    @profile
    def step(self, action):
        # time for carrier to take the action and repair one bicycle
        t = (np.abs(self.carrier_position % 3 - action % 3) + np.abs(self.carrier_position // 3 - action // 3)) / \
            self.R.loc[0].ride + self.repair
        t_cursor = self.T + t
        self.carrier_position = action
        reward = 0

        # update the atate of QN during the tansformation time
        while self.T < t_cursor:
            self.T = self.scheduler[0][0]
            if self.scheduler[0][1] == 1:
                # stat.loc[S], S = [scheduler[0][1], scheduler[0][2].a_id, T], S+1
                if self.scheduler[0][2].normal_bike == 0:
                    # this is a loss
                    reward -= 1
                    # loss.loc[L], L = [scheduler[0][2].a_id, self.T], L+1
                    event = [self.T + np.random.exponential(1 / self.R.loc[self.scheduler[0][2].a_id].cus_arr), 1,
                             self.scheduler[0][2]]
                    binaryInsert(self.scheduler, [event])
                else:
                    target = np.random.choice(np.arange(self.A + 1), 1, p=self.Q[self.scheduler[0][2].a_id])
                    if target == self.A:
                        # broken.loc[B], B = [self.scheduler[0][2].a_id, self.scheduler[0][2].normal_bike, self.scheduler[0][2].broken_bike, T], B+1
                        self.scheduler[0][2].move()
                        self.s[self.scheduler[0][2].a_id * 2], self.s[self.scheduler[0][2].a_id * 2 + 1] = \
                            self.scheduler[0][2].normal_bike, self.scheduler[0][2].broken_bike
                        continue
                    else:
                        self.scheduler[0][2].normal_bike -= 1
                        self.s[self.scheduler[0][2].a_id * 2] -= 1
                        event1 = [self.T + np.random.exponential(1 / self.R.loc[self.scheduler[0][2].a_id].ride), 2,
                                  self.a[target[0]]]
                        event2 = [self.T + np.random.exponential(1 / self.R.loc[self.scheduler[0][2].a_id].cus_arr), 1,
                                  self.scheduler[0][2]]
                        binaryInsert(self.scheduler, [event1, event2])
            else:
                # stat.loc[S], S = [scheduler[0][1], scheduler[0][2].a_id, T], S+1
                self.scheduler[0][2].normal_bike += 1
                self.s[self.scheduler[0][2].a_id * 2] += 1
            self.scheduler.pop(0)

        self.a[action].repair()
        s_ = self.s.copy()

        self.T = t_cursor
        if self.T < self.time_limit:
            return s_, reward, 0
        else:
            return s_, reward, 1