import torch as T
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from random import sample

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
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

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def store_memory(self, state, action, reward, state_, done):
        self.states = self.states.append(state)[-self.max_len:]
        self.actions = self.actions.append(action)[-self.max_len:]
        self.rewards = self.rewards.append(reward)[-self.max_len:]
        self.states_ = self.states_.append(state_)[-self.max_len:]
        self.dones = self.dones.append(done)[-self.max_len:]

    def get_memory_batch(self):
        state_batch = T.tensor(sample(self.states, self.batch_size)).to(self.device)
        action_batch = T.tensor(sample(self.actions, self.batch_size)).to(self.device)
        reward_batch = T.tensor(sample(self.rewards, self.batch_size)).to(self.device)
        _state_batch = T.tensor(sample(self.states_, self.batch_size)).to(self.device)
        done_batch = T.tensor(sample(self.dones, self.batch_size)).to(self.device)
        return state_batch, action_batch, reward_batch, _state_batch, done_batch

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.actor_network = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.LayerNorm(fc1_dims),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.LayerNorm(fc2_dims),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=1)
        )

        self.critic_network_1 = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.LayerNorm(fc1_dims),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims)
        )
        self.action_value = nn.Linear(n_actions, fc2_dims)
        self.critic_network_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        mu = self.actor_network(state)
        state_v = self.critic_network_1(state)
        action_v = self.action_value(action)
        state_action_v = self.critic_network_2(T.add(state_v, action_v))
        return mu, state_action_v

class Agent:
    def __init__(self, lr, tau, input_dims, n_actions, gamma=0.99, fc1_dims=400, fc2_dims=300,\
        batch_size=64, max_mem_len=50000):
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size

        self.actor_critic = ActorCriticNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions)
        self.target_actor_critic = ActorCriticNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions)
        
        self.memory = ReplayBuffer(max_len=max_mem_len, batch_size=batch_size)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_target_networks(tau=1)

    def choose_action(self, observation):
        self.actor_critic.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor_critic.device)
        mu, _ = self.actor_critic.forward(observation).to(actor_critic.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor_critic.device)
        self.actor_critic.train()
        return mu_prime.cpu().detach().numpy()

    def store_memory(self, state, action, reward, state_, done):
        return self.memory.store_memory(state, action, reward, state_, done)

    def learn(self):
        if len(self.memory.states) < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.memory.get_memory_batch()
        self.actor_critic.eval()
        self.target_actor_critic()

        target_actions, _ = self.target_actor_critic.forward(states)
        _, critic_value_ = self.target_critic.forward(states_, target_actions)
        actor_value, critic_value = self.actor_critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)
        
        target = reward + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.actor_critic.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        _, actor_loss = -self.actor_critic.forward(states, actor_value)
        actor_loss = T.mean(actor_loss)

        critic_value.backward()
        actor_value.backward()

        self.actor_critic.optimizer.step()

        self.update_target_networks()

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        ac_state_dict = self.actor_critic.state_dict()
        target_ac_state_dict = self.target_actor_critic.state_dict()

        for name in ac_state_dict:
            ac_state_dict = tau * ac_state_dict[name].clone() + \
                (1 - tau) * target_ac_state_dict[name].clone()
    
        self.target_actor_critic.load_state_dict(ac_state_dict)
    
