import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from NeuralNetwork import DQN

class Train():
    def __init__(
            self,
            n_actions,
            n_features,
            n_episodes,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_episodes = n_episodes
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self.target_net = DQN(n_features, n_actions)
        self.eval_net = DQN(n_features, n_actions)
        # self.eval_net.load_state_dict(torch.load('/content/params.pkl'))

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=self.lr)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval_net(Variable(torch.from_numpy(observation).float())).cpu().detach().numpy()
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            # self.sess.run(self.target_replace_op)
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        self.s = Variable(torch.from_numpy(batch_memory[:, :self.n_features]).float(), requires_grad=True)
        self.a = Variable(torch.from_numpy(batch_memory[:, self.n_features]).long())
        self.r = Variable(torch.from_numpy(batch_memory[:, self.n_features + 1]).float())
        self.s_ = Variable(torch.from_numpy(batch_memory[:, -self.n_features:]).float())

        current_Q_values = self.eval_net(self.s).gather(1, self.a.unsqueeze(1)).view(-1)
        next_Q_values = self.target_net(self.s_).detach().max(1)[0]
        # Compute the target of the current Q values
        target_Q_values = self.r + (self.gamma * next_Q_values)
        # Compute Bellman error
        loss = self.criterion(target_Q_values, current_Q_values)

        self.optimizer.zero_grad()
        # run backward pass
        loss.backward()

        # Perfom the update
        self.optimizer.step()

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


