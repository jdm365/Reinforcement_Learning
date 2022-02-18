import torch as T
import numpy as np
import torch.nn as nn

class Agent():
    def __init__(self, lr, gamma, n_actions, state_space, eps_start, eps_end, 
        eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.n_actions = n_actions

        self.state_space = state_space
        self.actions = [i for i in range(self.n_actions)]

        self.Q = {}

        self.init_Q()

    def init_Q(self):
        for state in self.state_space:
            for action in self.actions:
                self.Q[(state, action)] = 0.0
    
    def max_action(self, state):
        actions = np.array([self.Q[(state, a)] for a in self.actions])
        action = np.argmax(actions)
        return action

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.max_action(state)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_end \
            else self.eps_end
    
    def learn(self, state, action, reward, state_):
        a_max = self.max_action(state_)
        self.Q[(state, action)] = self.Q[(state, action)] + \
            self.lr * (reward + self.gamma * self.Q[(state_, a_max)] - \
            self.Q[(state, action)])

    