import torch as T
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from random import sample
import sys

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(\
            self.mu, self.sigma)

class ReplayBuffer:
    def __init__(self, max_len, batch_size):
        self.max_len = max_len
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.rewards = []
        self.states_ = []
        self.dones = []

    def store_memory(self, state, action, reward, state_, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states_.append(state_)
        self.dones.append(1-done)

        self.states = self.states[-self.max_len:]
        self.actions = self.actions[-self.max_len:]
        self.rewards = self.rewards[-self.max_len:]
        self.states_ = self.states_[-self.max_len:]
        self.dones = self.dones[-self.max_len:]

    def get_memory_batch(self):
        batch = np.random.choice(len(self.states), self.batch_size)

        state_batch = np.array(self.states)[batch]
        action_batch = np.array(self.actions)[batch]
        reward_batch = np.array(self.rewards)[batch]
        _state_batch = np.array(self.states_)[batch]
        done_batch = np.array(self.dones, dtype=np.float32)[batch]
        return state_batch, action_batch, reward_batch, _state_batch, done_batch

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, checkpoint_dir='Trained_Models/'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = checkpoint_dir + name

        self.actor_network = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            #nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            #nn.LayerNorm(fc2_dims),
            nn.ReLU()
        )
        self.actor_network.apply(self.init_weights)

        self.mu = nn.Linear(fc2_dims, n_actions)
        f3 = 3e-3
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.actor_network(state)
        mu = T.tanh(self.mu(x))
        return mu

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            f = 1./np.sqrt(layer.weight.data.shape[0])
            T.nn.init.uniform_(layer.weight, -f, f)
            T.nn.init.uniform_(layer.bias, -f, f)
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(self.checkpoint_file)


class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, checkpoint_dir='Trained_Models/'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = checkpoint_dir + name

        self.critic_network_1 = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            #nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            #nn.LayerNorm(fc2_dims),
            nn.ReLU(),
        )
        self.critic_network_1.apply(self.init_weights)

        self.action_value = nn.Linear(n_actions, fc2_dims)
        f3 = 3e-3
        T.nn.init.uniform_(self.action_value.weight.data, -f3, f3)
        T.nn.init.uniform_(self.action_value.bias.data, -f3, f3)

        self.q = nn.Linear(fc2_dims, 1)
        f4 = 3e-3
        T.nn.init.uniform_(self.q.weight.data, -f4, f4)
        T.nn.init.uniform_(self.q.bias.data, -f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_v = self.critic_network_1(state)
        action_v = F.relu(self.action_value(action))
        state_action_v = F.relu(T.add(state_v, action_v))
        return self.q(state_action_v)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            f = 1./np.sqrt(layer.weight.data.shape[0])
            T.nn.init.uniform_(layer.weight, -f, f)
            T.nn.init.uniform_(layer.bias, -f, f)
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(self.checkpoint_file)


class Agent:
    def __init__(self, lr_actor, lr_critic, tau, input_dims, n_actions, gamma=0.99, fc1_dims=400, fc2_dims=300,\
        batch_size=128, max_mem_len=50000):
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma

        self.actor = ActorNetwork(lr_actor, input_dims, fc1_dims, fc2_dims, n_actions, name='Actor')
        self.critic = CriticNetwork(lr_critic, input_dims, fc1_dims, fc2_dims, n_actions, name='Critic')
        self.target_actor = ActorNetwork(lr_actor, input_dims, fc1_dims, fc2_dims, n_actions, name='Target_Actor')
        self.target_critic = CriticNetwork(lr_critic, input_dims, fc1_dims, fc2_dims, n_actions, name='Target_Critic')

        self.memory = ReplayBuffer(max_len=max_mem_len, batch_size=batch_size)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_target_networks(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def store_memory(self, state, action, reward, state_, done):
        return self.memory.store_memory(state, action, reward, state_, done)

    def learn(self):
        if len(self.memory.states) < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.memory.get_memory_batch()

        states = T.tensor(states, dtype=T.float).to(self.critic.device)
        actions = T.tensor(actions, dtype=T.float).to(self.critic.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.critic.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.critic.device)
        dones = T.tensor(dones).to(self.critic.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)
        
        target = []
        for j in range(self.batch_size):
            target.append(rewards[j] + self.gamma*critic_value_[j]*dones[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        critic_loss = nn.MSELoss()
        self.critic.optimizer.zero_grad()
        critic_loss(target, critic_value).backward()
        self.critic.optimizer.step()

        actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_target_networks()

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

    def save_models(self):
        print('...Saving Models...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_models(self):
        print('...Loading Models...')
        self.actor.load_models()
        self.critic.load_models()
        self.target_actor.load_models()
        self.target_critic.load_models()
    
