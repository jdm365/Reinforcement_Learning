import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU()
        )
        self.mu = nn.Sequential(
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = 'cpu'#T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        observation = T.tensor(observation, dtype=T.float).to(self.device)
        x = self.shared_layers(observation)
        mu = Categorical(self.mu(x))
        v = self.v(x)
        return mu, v

    def save_actor_critic_dict(self, filename='Trained_Models/A2C_actor_critic'):
        T.save(self.state_dict(), filename)

    def load_actor_critic_dict(self, filename='Trained_Models/A2C_actor_critic'):
        self.load_state_dict(T.load(filename))

class Agent:
    def __init__(self, lr, n_actions, input_dims, \
        fc1_dims=256, fc2_dims=256, gamma=0.99):

        self.actor_critic = ActorCriticNetwork(lr, input_dims, fc1_dims, \
            fc2_dims, n_actions)
        self.gamma = gamma

    def choose_action(self, observation):
        mu, _ = self.actor_critic.forward(observation)
        action = mu.sample()
        log_probs = mu.log_prob(action)
        self.log_probs = log_probs
        return action.item()

    def learn(self, observation, reward, observation_, done):
        _, val_ = self.actor_critic.forward(observation_)
        _, val = self.actor_critic.forward(observation)

        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        advantage = reward + (self.gamma * val_ * (1-int(done))) - val
        actor_loss = -(advantage * self.log_probs)
        critic_loss = advantage.pow(2)

        total_loss = actor_loss + critic_loss

        self.actor_critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor_critic.optimizer.step()

    def save_models(self):
        print('...Saving Model...')
        self.actor_critic.save_actor_dict()

    def load_models(self):
        print('...Loading Models...')
        self.actor_critic.load_actor_dict()
