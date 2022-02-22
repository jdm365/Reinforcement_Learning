import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=0)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        observation = T.tensor(observation, dtype=T.float).to(self.device)
        mu = Categorical(self.actor(observation))
        return mu

    def save_actor_dict(self, filename):
        T.save(self.state_dict(), filename)

class Agent:
    def __init__(self, lr, input_dims,  n_actions, fc1_dims=256, fc2_dims=256, gamma=0.99):
        self.actor = ActorNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions)
        self.log_probs_memory = []
        self.reward_memory = []
        self.gamma = gamma

    def choose_action(self, observation):
        mu = self.actor.forward(observation)
        action = mu.sample()
        log_probs = mu.log_prob(action)
        self.log_probs_memory.append(log_probs)
        return action.item()

    def remember(self, reward):
        self.reward_memory.append(reward)
    
    def learn(self):
        discounted_rewards = []
        for t in range(len(self.reward_memory)):
            disc_rewards = 0
            gamma = 1
            for reward in self.reward_memory[t:]:
                disc_rewards += reward * gamma
                gamma *= self.gamma
            discounted_rewards.append(disc_rewards)
        
        discounted_rewards = T.tensor(discounted_rewards, dtype=T.float).to(self.actor.device)
        log_probs = T.tensor(self.log_probs_memory, dtype=T.float).to(self.actor.device)

        policy_loss = 0
        for return_estimate, log_probs in zip(discounted_rewards, self.log_probs_memory):
            policy_loss -= return_estimate * log_probs
        
        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

        self.log_probs_memory = []
        self.reward_memory = []

    def save_model(self, filename='Trained_Models/policy_gradient_actor'):
        print('...Saving Model...')
        self.actor.save_actor_dict(filename=filename)

    def load_model(self, filename='Trained_Models/policy_gradient_actor'):
        self.actor.load_state_dict(filename)