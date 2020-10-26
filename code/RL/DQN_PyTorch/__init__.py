from bikeNet import BikeNet
from model import Train
#from SimulationOptimization import BikeNet, Area, binaryInsert
import numpy as np
import random

#@profile
def run(env, RL):
    result = []
    for episode in range(RL.n_episodes):
        sum_r = 0
        step = 0
        # initial observation
        observation = env.warmup(RL)

        while True:

            # RL choose action based on observation
            action = RL.choose_action(observation)
            # action = (action+1)%9
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)

            if step % 10 == 0:
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
            sum_r += reward

        result.append([episode, sum_r])

    # end of game
    print('learning over')
    return result

if __name__ == '__main__':
    random.seed(0)
    N = 80  # total number of bikes in the QN
    A = 4  # A for areas, indicates the number of areas and the action space
    R = {}  # [customer_arrval, ride]
    for i in range(A): R[i] = [1.0 * i+0.5, 0.1]
    Q = [np.random.rand(A) for i in range(A)]
    Q = [q / sum(q) * 0.9 for q in Q]
    Q = [np.append(q, 0.1) for q in Q]
    # Q = [[0,0.9,0.1], [0.9,0,0.1]]
    t_repair = 2
    warmup_time = 500
    run_time = 180

    env = BikeNet(N=200,
                  A=4,
                  R=R,
                  Q=Q,
                  repair=t_repair,
                  warmup_time=warmup_time,
                  run_time=run_time,
                  start_position=0)

    RL = Train(n_actions=A,
               n_features=2 * A,
               n_episodes=12,
               learning_rate=0.001,
               reward_decay=0.9,
               e_greedy=0.8,
               replace_target_iter=200,
               memory_size=2000,
               # output_graph=True
               )
    output = run(env, RL)
    print(output)
    # RL.plot_cost()