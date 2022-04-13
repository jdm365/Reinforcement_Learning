import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
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

        if len(self.states) == self.max_len:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.states_.pop(0)
            self.dones.pop(0)

    def get_memory_batch(self):
        batch = np.random.choice(len(self.states), self.batch_size)

        state_batch = np.array(self.states)[batch]
        action_batch = np.array(self.actions)[batch]
        reward_batch = np.array(self.rewards)[batch]
        _state_batch = np.array(self.states_)[batch]
        done_batch = np.array(self.dones, dtype=np.float32)[batch]
        return state_batch, action_batch, reward_batch, _state_batch, done_batch


class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, 
        checkpoint_dir='Trained_Models/'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = checkpoint_dir + name
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc1_norm = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc2_norm = nn.LayerNorm(fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        fc1_out = F.relu(self.fc1_norm(self.fc1(state)))
        fc2_out = F.relu(self.fc2_norm(self.fc2(fc1_out)))
        mu = T.tanh(self.mu(fc2_out))
        return mu

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(self.checkpoint_file)


class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, 
        checkpoint_dir='Trained_Models/'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = checkpoint_dir + name

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc1_norm = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc2_norm = nn.LayerNorm(fc2_dims)
        self.action_value = nn.Linear(n_actions, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        fc1_out = F.relu(self.fc1_norm(self.fc1(state)))
        fc2_out = F.relu(self.fc2_norm(self.fc2(fc1_out)))

        action_v = self.action_value(action)
        state_action_v = F.relu(T.add(fc2_out, action_v))
        return self.q(state_action_v)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(self.checkpoint_file)


class Agent:
    def __init__(self, lr_actor, lr_critics, tau, input_dims, n_actions, fc1_dims, \
        fc2_dims, env, gamma=0.99, batch_size=256, max_mem_len=50000, learn_freq=4):
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_iters = 0
        self.learn_freq = learn_freq
        self.env = env

        self.actor = ActorNetwork(lr_actor, input_dims, fc1_dims, fc2_dims, \
            n_actions, name='actor')
        self.target_actor = ActorNetwork(lr_actor, input_dims, fc1_dims, fc2_dims, \
            n_actions, name='target_actor')

        self.critic_1 = CriticNetwork(lr_critics, input_dims, fc1_dims, fc2_dims, \
            n_actions, name='critic_1')
        self.target_critic_1 = CriticNetwork(lr_critics, input_dims, fc1_dims, fc2_dims, \
            n_actions, name='target_critic_1')

        self.critic_2 = CriticNetwork(lr_critics, input_dims, fc1_dims, fc2_dims, \
            n_actions, name='critic_2')
        self.target_critic_2 = CriticNetwork(lr_critics, input_dims, fc1_dims, fc2_dims, \
            n_actions, name='target_critic_2')

        self.memory = ReplayBuffer(max_len=max_mem_len, batch_size=batch_size)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_target_networks(tau=1)

    def norm_action(self, action):
        high = self.env.action_space.high
        low = self.env.action_space.low
        if type(action) == T.Tensor:
            high = T.tensor(high).to(self.critic_1.device)
            low = T.tensor(low).to(self.critic_1.device)
            normalized = T.clamp(action, low, high)
        else:
            normalized = np.clip(action, low, high)
        return normalized

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        mu_prime = self.norm_action(mu_prime.cpu().detach().numpy())
        self.actor.train()
        return mu_prime

    def store_memory(self, state, action, reward, state_, done):
        return self.memory.store_memory(state, action, reward, state_, done)

    def learn(self):
        if len(self.memory.states) < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.memory.get_memory_batch()

        states = T.tensor(states, dtype=T.float).to(self.critic_1.device)
        actions = T.tensor(actions, dtype=T.float).to(self.critic_1.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.critic_1.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.critic_1.device)
        dones = T.tensor(dones).to(self.critic_1.device)

        self.target_actor.eval()
        self.target_critic_1.eval()
        self.critic_1.eval()
        self.target_critic_2.eval()
        self.critic_2.eval()

        target_actions = self.norm_action(self.target_actor.forward(states_))
        
        critic_1_value_ = self.target_critic_1.forward(states_, target_actions)
        critic_1_value = self.critic_1.forward(states, actions)

        critic_2_value_ = self.target_critic_2.forward(states_, target_actions)
        critic_2_value = self.critic_2.forward(states, actions)
        
        target = []
        for j in range(self.batch_size):
            val_1 = rewards[j] + self.gamma*critic_1_value_[j]*dones[j]
            val_2 = rewards[j] + self.gamma*critic_2_value_[j]*dones[j]
            target.append(T.min(val_1, val_2))
        target = T.tensor(target).to(self.critic_1.device)
        target = target.view(self.batch_size, 1)

        self.critic_1.train()
        self.critic_2.train()
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_1_loss = F.mse_loss(target, critic_1_value).to(self.critic_1.device)
        critic_2_loss = F.mse_loss(target, critic_2_value).to(self.critic_1.device)
        (critic_1_loss + critic_2_loss).backward()
        #nn.utils.clip_grad_norm_(self.critic.parameters(), 1e-2)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_iters += 1

        if self.learn_iters % self.learn_freq == 0:
            self.critic_1.eval()
            self.critic_2.eval()
            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic_1.forward(states, self.actor.forward(states)).mean()
            self.actor.train()
            
            actor_loss.backward()
            #nn.utils.clip_grad_norm_(self.actor.parameters(), 1e-2)
            self.actor.optimizer.step()

            self.update_target_networks()

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_dict = self.actor.state_dict()
        target_critic_1_dict = self.critic_1.state_dict()
        target_critic_2_dict = self.critic_2.state_dict()

        for name, param in self.actor.state_dict().items():
            target_actor_dict[name] = tau * param.data + (1-tau) * target_actor_dict[name].clone()
        
        for name, param in self.critic_1.state_dict().items():
            target_critic_1_dict[name] = tau * param.data + (1-tau) * target_critic_1_dict[name].clone()

        for name, param in self.critic_1.state_dict().items():
            target_critic_2_dict[name] = tau * param.data + (1-tau) * target_critic_2_dict[name].clone()

        self.target_actor.load_state_dict(target_actor_dict)
        self.target_critic_1.load_state_dict(target_critic_1_dict)
        self.target_critic_2.load_state_dict(target_critic_2_dict)


    def save_models(self):
        print('...Saving Models...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
    
    def load_models(self):
        print('...Loading Models...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
    