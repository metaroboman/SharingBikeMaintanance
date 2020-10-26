import heapq
import random
import numpy as np


class BikeNet():
    def __init__(self, N, A, R, Q, repair, warmup_time, run_time, start_position=0):
        self.N = N
        self.A = A
        self.R = R
        self.Q = Q
        self.repair = repair
        self.warmup_time = warmup_time
        self.run_time = run_time
        self.time_limit = warmup_time + run_time
        self.car = start_position
        self.edge = int(self.A**0.5)
        self.areas = list(range(A + 1))

        self.reset()

    def reset(self):
        self.T = 0
        self.carrier_position = self.car
        self.state = [int(self.N / self.A)] * self.A + [0] * self.A
        self.scheduler = []
        heapq.heapify(self.scheduler)
        for i in range(self.A):
            heapq.heappush(self.scheduler, [random.expovariate(self.R[i][0]), -1, i])
        heapq.heapify(self.scheduler)
        return np.array(self.state.copy())

    def warmup(self, rl):
        s = self.reset()

        while self.T < self.warmup_time:
            action = random.randint(0,3)
            result = self.step(action)
            s_, r = result[0], result[1]
            rl.store_transition(s, action, r, s_)
            s = s_

        self.T = self.warmup_time
        return np.array(s.copy())


    def step(self, action):
        rewards = 0
        # time for carrier to take the action and repair one bicycle
        t = (abs(self.carrier_position % (self.edge) - action % (self.edge)) + abs(
            self.carrier_position // (self.edge) - action // (self.edge))) * 0.5
        if self.state[action + self.A] > 0:
            t_cursor = self.T + t + self.repair
        else:
            t_cursor = self.T + t

        event = self.scheduler[0]
        self.T, kind, location = event[0], event[1], event[2]

        # update the atate of QN during the tansformation time
        while self.T < t_cursor:
            # 车到达
            if kind == 1:
                self.state[location] += 1
                heapq.heappop(self.scheduler)
            else:
                # 顾客到达
                if self.state[location] == 0:  # 但没车
                    rewards -= 1
                    heapq.heappop(self.scheduler)
                else:
                    target = np.random.choice(self.areas, 1, p=self.Q[location])[0]
                    if target == self.A:  # 顾客到达，发现是坏车
                        self.state[location] -= 1
                        self.state[location + self.A] += 1
                        continue
                    else:  # 顾客到达，顺利骑行
                        self.state[location] -= 1
                        heapq.heappop(self.scheduler)
                        next_time = random.expovariate(self.R[location][1]) + self.T
                        if next_time <= self.time_limit:
                            heapq.heappush(self.scheduler, [next_time, 1, target])
                next_time = random.expovariate(self.R[location][0]) + self.T
                if next_time <= self.time_limit:
                    heapq.heappush(self.scheduler, [next_time, -1, location])

            if self.scheduler:
                event = self.scheduler[0]
                self.T, kind, location = event[0], event[1], event[2]
            else:
                break

        # if self.state[action + self.A] > 0:
        #     self.state[self.carrier_position] += 1
        #     self.state[self.carrier_position + self.A] -= 1

        self.carrier_position = action
        self.T = t_cursor

        s_ = np.array(self.state.copy())

        if self.T <= self.time_limit and self.scheduler:
            return s_, rewards, 0
        else:
            return s_, rewards, 1

# if __name__ == '__main__':
#     random.seed(0)
#     N = 80  # total number of bikes in the QN
#     A = 4  # A for areas, indicates the number of areas and the action space
#     R = {}  # [customer_arrval, ride]
#     for i in range(A): R[i] = [1.0 * i+0.5, 0.1]
#     Q = [np.random.rand(A) for i in range(A)]
#     Q = [q / sum(q) * 0.9 for q in Q]
#     Q = [np.append(q, 0.1) for q in Q]
#     # Q = [[0,0.9,0.1], [0.9,0,0.1]]
#     t_repair = 2
#     warmup_time = 500
#     run_time = 180
#
#     env = BikeNet(N=200,
#                   A=4,
#                   R=R,
#                   Q=Q,
#                   repair=t_repair,
#                   warmup_time=warmup_time,
#                   run_time=run_time,
#                   start_position=0)
#     RL = Train(n_actions=A,
#                n_features=2 * A,
#                n_episodes=12,
#                learning_rate=0.001,
#                reward_decay=0.9,
#                e_greedy=0.8,
#                replace_target_iter=200,
#                memory_size=2000,
#                # output_graph=True
#                )
#     print(env.warmup(RL))
#     print(RL.memory)
    # r = 0
    # while env.T<680:
    #     r += env.step(random.randint(0, 3))[1]
    # print(env.state)
    # print(env.c)
    # print(env.T)
    # # print(env.step(1))
    # print(env.scheduler)
