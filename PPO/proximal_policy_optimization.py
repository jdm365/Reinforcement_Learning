import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from utils import init_linear


class ReplayBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.rewards = []
        self.states_ = []
        self.vals = []
        self.probs = []
        self.dones = []

    def remember(self, state, action, reward, state_, val, probs, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states_.append(state_)
        self.vals.append(val)
        self.probs.append(probs)
        self.dones.append(done)

    def get_batch(self):
        index = np.random.randint(0, len(self.states) - self.batch_size)

        states = np.array(self.states)[index:index+self.batch_size]
        actions = np.array(self.actions)[index:index+self.batch_size]
        rewards = np.array(self.rewards)[index:index+self.batch_size]
        states_ = np.array(self.states_)[index:index+self.batch_size]
        vals = np.array(self.vals)[index:index+self.batch_size]
        probs = np.array(self.probs)[index:index+self.batch_size]
        dones = np.array(self.dones)[index:index+self.batch_size]
        return states, actions, rewards, states_, vals, probs, dones
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.states_ = []
        self.vals = []
        self.probs = []
        self.dones = []


class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, fc1_dims, fc2_dims):
        super(ActorCriticNetwork, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU()
        )
        init_linear(self.shared_layers)

        self.actor_network = nn.Sequential(
            self.shared_layers,
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic_network = nn.Sequential(
            self.shared_layers,
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        pi = self.actor_network(state)
        v = self.critic_network(state)
        return pi, v

class Agent:
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, batch_size=256, \
        horizon=512, n_updates=10, eta=0.35, gamma=0.99, gae_lambda=0.90):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(lr, n_actions, input_dims, \
            fc1_dims, fc2_dims)
        self.log_prob = None
        self.memory = ReplayBuffer(batch_size=batch_size)
        self.n_updates = n_updates
        self.batch_size = batch_size
        self.eta = eta
        self.horizon = horizon
        self.gae_lambda = gae_lambda
        self.steps_taken = 0

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor_critic.device)
        self.actor_critic.eval()
        probs, value = self.actor_critic.forward(state)
        action_probs = Categorical(probs)
        action = action_probs.sample()
        self.actor_critic.train()

        self.steps_taken += 1
        if self.steps_taken % self.horizon == 0:
            self.learn()
        return action.item(), action_probs.log_prob(action).item(), value.item()

    def remember(self, state, action, reward, state_, value, probs, done):
        self.memory.remember(state, action, reward, state_, value, probs, done)

    def learn(self):
        if self.steps_taken < self.batch_size:
            return

        for _ in range(self.n_updates):
            states, actions, rewards, states_, vals, probs, dones = self.memory.get_batch()

            states = T.tensor(states, dtype=T.float).to(self.actor_critic.device)
            actions = T.tensor(actions, dtype=T.float).to(self.actor_critic.device)
            rewards = T.tensor(rewards, dtype=T.float).to(self.actor_critic.device)
            states_ = T.tensor(states_, dtype=T.float).to(self.actor_critic.device)
            vals = T.tensor(vals, dtype=T.float).to(self.actor_critic.device)
            probs = T.tensor(probs, dtype=T.float).to(self.actor_critic.device)
            dones = T.tensor(dones, dtype=T.float).to(self.actor_critic.device)

            advantages = []
            for t in range(self.batch_size):
                discount = 1
                A = 0
                for k in range(t, self.batch_size-1):
                    A += discount * (rewards[k] + self.gamma * vals[k+1])*(1-int(dones[k]) \
                        - vals[k])
                    discount *= self.gamma * self.gae_lambda
                advantages.append(A)
            advantages = T.tensor(advantages, dtype=T.float).to(self.actor_critic.device)

            self.actor_critic.eval()
            pi, new_vals = self.actor_critic.forward(states)
            self.actor_critic.train()
            dist = Categorical(pi)

            old_probs = probs
            new_probs = dist.log_prob(actions)

            probs_ratio = (new_probs - old_probs).exp()  ## (a/b) = log(a/b).exp() = (log(a) - log(b)).exp()
            clamped_ratio = probs_ratio.clamp(1 - self.eta, 1 + self.eta)


            actor_loss = -T.min((probs_ratio * advantages), (clamped_ratio * advantages))
            critic_loss = (advantages + (vals - new_vals))**2
            total_loss = actor_loss.mean() + 0.5 * critic_loss.mean()

            self.actor_critic.optimizer.zero_grad()
            total_loss.backward()
            self.actor_critic.optimizer.step()
        self.memory.clear_memory()