import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()

        self.actor_network = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-3)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        observation = T.tensor(observation, dtype=T.float).to(self.device)
        mu = Categorical(self.actor_network(observation))
        return mu

    def save_actor_dict(self, filename='Trained_Models/A2C_actor'):
        T.save(self.state_dict(), filename)

    def load_actor_dict(self, filename='Trained_Models/A2C_actor'):
        self.load_state_dict(T.load(filename))

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()

        self.critic_network = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-3)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        observation = T.tensor(observation, dtype=T.float).to(self.device)
        return self.critic_network(observation)

    def save_critic_dict(self, filename='Trained_Models/A2C_critic'):
        T.save(self.state_dict(), filename)

    def load_critic_dict(self, filename='Trained_Models/A2C_critic'):
        self.load_state_dict(T.load(filename))


class Agent:
    def __init__(self, lr_actor, lr_critic, n_actions, input_dims, \
        fc1_dims=256, fc2_dims=256, gamma=0.99, max_mem_length=5):

        self.actor = ActorNetwork(lr_actor, input_dims, fc1_dims, \
            fc2_dims, n_actions)
        self.critic = CriticNetwork(lr_critic, input_dims, fc1_dims, \
            fc2_dims, n_actions)
        self.gamma = gamma
        self.max_mem_length = max_mem_length

        self.state_memory = []
        self.log_probs_memory = []
        self.reward_memory = []

    def choose_action(self, observation):
        mu = self.actor.forward(observation)
        action = mu.sample()
        log_probs = mu.log_prob(action)
        self.log_probs_memory.append(log_probs)
        return action.item()

    def remember(self, observation, reward):
        self.state_memory.append(observation)
        self.reward_memory.append(reward)

    def learn(self, observation_, done):
        R = 0
        gamma = 1
        for reward in self.reward_memory[1:]:
            R += gamma * reward
            gamma *= self.gamma
        val = self.critic.forward(self.state_memory[0])
        val_ = self.critic.forward(observation_)*(1-int(done))
        advantage = R + gamma * val_ - val
        log_probs = self.log_probs_memory[0]

        actor_loss = -(advantage * log_probs)
        critic_loss = advantage.pow(2)

        total_loss = (actor_loss + critic_loss)

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

        self.log_probs_memory = []
        self.state_memory = []
        self.reward_memory = []

    def save_models(self):
        print('...Saving Model...')
        self.actor.save_actor_dict()
        self.critic.save_critic_dict()

    def load_models(self):
        print('...Loading Models...')
        self.actor.load_actor_dict()
        self.critic.load_critic_dict()
