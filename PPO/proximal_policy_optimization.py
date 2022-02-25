import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from utils import init_linear
import sys


class ReplayBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.rewards = []
        self.vals = []
        self.probs = []
        self.dones = []

    def remember(self, state, action, reward, val, probs, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.vals.append(val)
        self.probs.append(probs)
        self.dones.append(done)

    def get_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.rewards),\
                np.array(self.vals),\
                np.array(self.probs),\
                np.array(self.dones),\
                batches
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.vals = []
        self.probs = []
        self.dones = []


class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, fc1_dims, fc2_dims):
        super(ActorCriticNetwork, self).__init__()

        self.actor_network = nn.Sequential(
            nn.Linear(*input_dims, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        init_linear(self.actor_network)

        self.critic_network = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, 1)
        )
        init_linear(self.critic_network)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = 'cpu'#T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        pi = self.actor_network(state)
        v = self.critic_network(state)
        return pi, v

class Agent:
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, batch_size=8, \
        horizon=32, n_updates=4, eta=0.2, gamma=0.99, gae_lambda=0.95):
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
        self.early_stop = 0.15

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor_critic.device)
        pi, value = self.actor_critic.forward(state)
        action_probs = Categorical(pi)
        action = action_probs.sample()

        self.steps_taken += 1
        return action.item(), action_probs.log_prob(action).item(), value.item()

    def remember(self, state, action, reward, value, probs, done):
        self.memory.remember(state, action, reward, value, probs, done)

    def calc_advantages(self, rewards, vals, dones):
        advantages = []
        for t in range(len(rewards)):
            discount = 1
            A = 0
            for k in range(t, len(rewards)-1):
                terminal = 1 - int(dones[k])
                delta = rewards[k] + self.gamma * vals[k+1] * terminal - vals[k]
                A += discount * delta
                discount *= self.gamma * self.gae_lambda
                if terminal:
                    break
            advantages.append(A)
        advantages = np.array(advantages, dtype=np.float)
        return advantages

    def learn(self):
        if self.steps_taken < self.batch_size:
            return

        for _ in range(self.n_updates):
            states, actions, rewards, vals, probs, dones, batches = self.memory.get_batches()
            advantages = self.calc_advantages(rewards, vals, dones)
            values = T.tensor(vals, dtype=T.float).to(self.actor_critic.device)
            advantages = T.tensor(advantages, dtype=T.float).squeeze().to(self.actor_critic.device)

            for batch in batches:
                state_batch = T.tensor(states[batch], dtype=T.float).to(self.actor_critic.device)
                action_batch = T.tensor(actions[batch], dtype=T.float).to(self.actor_critic.device)
                probs_batch = T.tensor(probs[batch], dtype=T.float).to(self.actor_critic.device)

                pi, new_vals = self.actor_critic.forward(state_batch)
                dist = Categorical(pi)
                old_probs = probs_batch
                new_probs = dist.log_prob(action_batch)

                probs_ratio = (new_probs - old_probs).exp()  ## (a/b) = log(a/b).exp() = (log(a) - log(b)).exp()
                clamped_ratio = probs_ratio.clamp(1 - self.eta, 1 + self.eta)

                #if T.sum((new_probs - old_probs) * new_probs.exp()) > self.early_stop:
                #    return
                actor_loss = -T.min(probs_ratio * advantages[batch], clamped_ratio * advantages[batch]).mean()
                critic_loss = T.mean((advantages[batch] + (values[batch] - new_vals.squeeze())).pow(2))
                total_loss = actor_loss + 0.5 * critic_loss

                self.actor_critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor_critic.optimizer.step()
        self.memory.clear_memory()