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
        self.device = 'cpu'#T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        observation = T.tensor(observation, dtype=T.float).to(self.device)
        mu = Categorical(self.actor(observation))
        return mu

    def save_actor_dict(self, filename='Trained_Models/A2C_actor'):
        T.save(self.state_dict(), filename)

    def load_actor_dict(self, filename='Trained_Models/A2C_actor'):
        self.load_state_dict(T.load(filename))

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        observation = T.tensor(observation, dtype=T.float).to(self.device)
        value = self.value(observation)
        return value

    def save_critc_dict(self, filename='Trained_Models/A2C_critic'):
        T.save(self.state_dict(), filename)
    
    def load_critc_dict(self, filename='Trained_Models/A2C_critic'):
        self.load_state_dict(T.load(filename))

class Agent:
    def __init__(self, lr_actor, lr_critic, n_actions, input_dims, \
        fc1_dims=256, fc2_dims=256, gamma=0.99):

        self.actor = ActorNetwork(lr_actor, input_dims, fc1_dims, \
            fc2_dims, n_actions)
        self.critic = CriticNetwork(lr_critic, input_dims, fc1_dims, \
            fc2_dims, n_actions)
        
        self.state_memory = []
        self.reward_memory = []
        self.log_probs_memory = []
        self.gamma = gamma

    def choose_action(self, observation):
        mu = self.actor.forward(observation)
        action = mu.sample()
        log_probs = mu.log_prob(action)
        self.log_probs_memory.append(log_probs)
        return action.item()

    def remember(self, observation, reward):
        self.state_memory.append(observation)
        self.reward_memory.append(reward)

    def learn(self):
        states = T.tensor(self.state_memory, dtype=T.float).to(self.critic.device)
        critic_vals = self.critic.forward(states)

        actor_loss = 0
        critic_loss = 0
        for t in range(len(self.state_memory)-1):
            advantage = self.reward_memory[t+1] + self.gamma * critic_vals[t+1] - critic_vals[t]
            actor_loss -= advantage * self.log_probs_memory[t]
            critic_loss += 0.5 * advantage ** 2

        actor_loss = actor_loss / (len(self.state_memory) - 1)
        actor_loss.to(self.actor.device)
        critic_loss = critic_loss / (len(self.state_memory) - 1)
        critic_loss.to(self.critic.device)

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        critic_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
        self.critic.optimizer.step()

        self.state_memory = []
        self.log_probs_memory = []
        self.reward_memory = []

    def save_models(self):
        print('...Saving Model...')
        self.actor.save_actor_dict(filename=filename)
        self.critic.save_actor_dict(filename=filename)

    def load_models(self):
        print('...Loading Models...')
        self.actor.load_actor_dict()
        self.critic.load_critic_dict()
