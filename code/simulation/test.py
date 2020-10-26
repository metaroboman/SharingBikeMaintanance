import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import random as random
import csv
import heapq

# initialize parameters
rate = {'lambda0' : 10.0,
        'lambda1' : 10.0,
        'lambda2' : 8.0,
        'lambda3' : 12.0,
        'car_move': 1.0,
        'gamma' : 1.0,
        'phi' : 5.0,
        'delta' : 1.0,
        'broken' : 0.2
       }
#queues: A0, A1, C, F, D, R00, R01, R10, R11
pij = [[0.1, 0.2, 0.3, 0.4],
       [0.2, 0.1, 0.3, 0.4],
       [0.3, 0.2, 0.1, 0.4],
       [0.4, 0.2, 0.3, 0.1],]


# the influence of phi
# in this version, carrier moves among areas to collect broken bikes
# then brings them to the repair center, and bring the normal bikes back to areas
class BikeNet():
    def __init__(self, N, A, R, P, warmup_time, run_time):
        self.N = N
        self.A = A
        self.R = R
        self.P = P
        self.warmup_time = warmup_time
        self.run_time = run_time
        self.time_limit = warmup_time + run_time
        # self.C = 0
        self.F = 0
        # self.D = 0
        self.capacity = 12
        self.areas = list(range(A))

        self.serverd_customers = 0

    def reset(self):
        self.T = 0
        #         self.C = 0
        self.F = 0
        #         self.D = 0
        self.serverd_customers = 0
        # queues: A0, A1, B0, B1, FN, FB, R00, R01, R10, R11
        self.state = [int(self.N / self.A)] * self.A + [0] * (self.A ** 2 + 2 + self.A)
        self.scheduler = []
        heapq.heapify(self.scheduler)
        for i in range(self.A):
            # state: [time, type, start, terminal]
            heapq.heappush(self.scheduler, [random.expovariate(self.R['lambda' + str(i)]), -1, i, 0])
        # event of the carrier, [time, kind, place, dummy parameter]
        heapq.heappush(self.scheduler, [0, 0, 0, [0, 0]])
        # return self.state + [self.T]

    def simulate(self):
        self.reset()
        while self.T <= self.time_limit:
            self.step()
        return self.serverd_customers / (self.T - 10000)

    def get_rho(self, path):
        s, t = int(path[0]), int(path[1])
        if s == t:
            return 0.5
        elif abs(s - t) == 2:
            return 2.0
        else:
            return 1.0

    def get_index(self, target):
        if target == 'c':
            return 4
        elif target == 'f':
            return 5
        elif target == 'd':
            return 6
        else:
            s, t = int(target[0]), int(target[1])
            return 2 * self.A + 2 + 4 * s + t

    def add_event(self, kind, s):
        if kind == 2:
            next_time = random.expovariate(self.R['gamma']) + self.T
            start, end = s, 'f'
        elif kind == 3:
            next_time = random.expovariate(self.R['phi']) + max(self.T, self.F)
            self.F, start, end = next_time, 'f', 'd'
        elif kind == 0:
            next_time = random.expovariate(self.R['delta']) + self.T
            start, end = random.choice(list(range(self.A))), [s, 0]
        heapq.heappush(self.scheduler, [next_time, kind, start, end])

    def step(self):

        event = self.scheduler[0]
        self.T, kind, start, terminal = event[0], event[1], event[2], event[3]

        '''kind:
        -1: customer ride a bike away
         0: carrier arrives as a area
         1: a bike arrives
         2: carrier full of broken bikes arrives as the repairing center
         3: a bike was fixed
        '''
        if kind == 0:  # carrier 正在areas之间逡巡中，鱼戏莲叶南，鱼戏莲叶北
            normal, broken = terminal[0], terminal[1]
            if normal > 0:  # 如果有好车，先把好车放下
                number = min(normal, int(self.capacity/self.A))
                self.state[start] += number
                self.scheduler[0][3][0] -= number
                #normal -= number
            if self.state[self.A + start] > 0 and broken < self.capacity:  # 如果有坏车，并且运载车没满，把坏车装上运载车
                number = min(self.capacity - broken, self.state[self.A + start])
                self.state[self.A + start] -= number
                self.scheduler[0][3][1] += number
                broken += number
            if broken == self.capacity:
                heapq.heappop(self.scheduler)
                self.add_event(2, start)
            else:
                self.scheduler[0][0] += random.expovariate(self.R['car_move'])
                self.scheduler[0][2] = (start + 1) % 4
                heapq.heapify(self.scheduler)
        elif kind == 1:  # 顾客骑行到达
            self.state[self.get_index(start)] -= 1
            if random.random() < self.R['broken']:
                self.state[self.A + terminal] += 1
                heapq.heappop(self.scheduler)
            else:
                self.state[terminal] += 1
                heapq.heappop(self.scheduler)
        elif kind == 2:
            if self.state[self.A * 2 + 1]==0:
                self.add_event(3, start)
            self.state[self.A * 2 + 1] += self.capacity
            heapq.heappop(self.scheduler)
            number = min(self.capacity, self.state[self.A * 2])
            self.add_event(0, number)
            self.state[self.A * 2] -= number
        elif kind == 3:
            self.state[self.A * 2] += 1
            self.state[self.A * 2 + 1] -= 1
            heapq.heappop(self.scheduler)
            if self.state[self.A * 2 + 1] > 0:
                self.add_event(3)
        else:  # 顾客到达
            if self.state[start] == 0:  # 但没车
                heapq.heappop(self.scheduler)
                next_time = random.expovariate(self.R['lambda' + str(start)]) + self.T
                heapq.heappush(self.scheduler, [next_time, -1, start, 0])
            else:
                target = str(start) + str(np.random.choice(self.areas, 1, p=self.P[start])[0])
                if self.T > 0: self.serverd_customers += 1
                self.state[start] -= 1
                self.state[self.get_index(target)] += 1
                heapq.heappop(self.scheduler)
                next_time = random.expovariate(self.get_rho(target)) + self.T
                heapq.heappush(self.scheduler, [next_time, 1, target, int(target[1])])
                next_time = random.expovariate(self.R['lambda' + str(start)]) + self.T
                heapq.heappush(self.scheduler, [next_time, -1, start, 0])

        # return self.state+[self.T]

if __name__ == '__main__':
    random.seed(1)
    N = 100  # total number of bikes in the QN
    A = 4  # A for areas, indicates the number of areas and the action space
    R = rate
    P = pij
    warmup_time = 0

    run_time = 20000

    env = BikeNet(N=N,
                  A=A,
                  R=R,
                  P=P,
                  # repair=t_repair,
                  warmup_time=warmup_time,
                  run_time=run_time)
    # start_position=0)

    result = []
    for i in range(1, 3):
        env.R['phi'] = i * 0.1
        result.append(env.simulate())

    # plt.plot(result)
    # plt.show()