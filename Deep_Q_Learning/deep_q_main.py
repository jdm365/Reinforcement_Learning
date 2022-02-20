import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, fc1_dims, fc2_dims):
        super(ActorCriticNetwork, self).__init__()
        self.actor_network = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax()
        )

        self.critic_network = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
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
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, \
        gamma=0.99):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(lr, n_actions, input_dims, \
            fc1_dims, fc2_dims)
        self.log_prob = None

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
        probs, _ = self.actor_critic.forward(state)
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        self.log_prob = action_probs.log_prob(action)
        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor([reward], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value
        actor_loss = -self.log_prob * delta
        critic_loss = delta ** 2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()


    