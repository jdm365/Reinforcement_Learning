import gym
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm

class Memory:
    def __init__(self, n_steps=100, batch_size=64):
        self.n_steps = n_steps
        self.batch_size = batch_size

        self.real_observations = []
        self.real_actions = []
        self.real_rewards = []
        self.real_observations_ = []
        self.real_dones = []

        self.imag_observations = []
        self.imag_actions = []
        self.imag_rewards = []
        self.imag_observations_ = []
        self.imag_dones = []
        self.imag_log_probs = []
        self.imag_vals = []

    def remember_real(self, obs, act, rew, obs_, done):
        self.real_observations.append(obs)
        self.real_actions.append(act)
        self.real_rewards.append(rew)
        self.real_observations_.append(obs_)
        self.real_dones.append(done)

    def remember_imaginary(self, obs, act, rew, obs_, done, vals=None):
        self.imag_observations.append(obs)
        self.imag_actions.append(act)
        self.imag_rewards.append(rew)
        self.imag_observations_.append(obs_)
        self.imag_dones.append(done)
        if vals != None:
            self.imag_vals.append(vals)

    def remember_log_probs(self, log_probs):
        self.imag_log_probs.append(log_probs)

    def get_real_data(self):
        index = np.random.randint(0, len(self.real_dones), self.batch_size)

        obs = np.array(self.real_observations)[index]
        act = np.array(self.real_actions)[index]
        rew = T.FloatTensor(self.real_rewards)[index]
        obs_ = T.FloatTensor(self.real_observations_)[index]
        dones = T.FloatTensor(self.real_dones)[index]

        return obs, act, rew, obs_, dones

    def get_imag_data(self, inner=False):
        if inner:
            rew = self.imag_rewards
            log_probs = self.imag_log_probs
            self.reset_imag_memory()
            return rew, log_probs

        batch_start = np.arange(0, len(self.imag_observations), self.batch_size)
        indices = np.arange(len(self.imag_observations), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        obs = np.array(self.imag_observations)
        act = T.FloatTensor(self.imag_actions)
        rew = T.FloatTensor(self.imag_rewards)
        obs_ = T.FloatTensor(self.imag_observations_)
        dones = T.FloatTensor(self.imag_dones)
        vals = T.FloatTensor(self.imag_vals)
        log_probs = T.FloatTensor(self.imag_log_probs)

        self.reset_imag_memory()
        return obs, act, rew, obs_, dones, log_probs, vals, batches

    def reset_imag_memory(self):
        self.imag_observations = []
        self.imag_actions = []
        self.imag_rewards = []
        self.imag_observations_ = []
        self.imag_dones = []
        self.imag_log_probs = []
        self.imag_vals = []


class DynamicsModel(nn.Module):
    def __init__(self, lr, obs_dims, act_dims, fc1_dims, fc2_dims):
        super(DynamicsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dims+act_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, obs_dims+2)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation, action):
        obs = T.tensor(observation, dtype=T.float).to(self.device)
        act = T.tensor(action, dtype=T.float).to(self.device)
        act = act.reshape(*act.shape, 1)
        input = T.cat((obs, act), dim=-1)
        output = self.model(input)
        if len(output.shape) == 1:
            obs_, rew, done = output[:-2], output[-2], output[-1]
            return obs_, rew, done.long()
        obs_, rew, done = output[:, :-2], output[:, -2], output[:, -1]
        return obs_, rew, done.long()

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()

        self.actor_network = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        obs = T.tensor(observation, dtype=T.float).to(self.device)
        return self.actor_network(obs)

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims):
        super(CriticNetwork, self).__init__()

        self.critic_network = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        obs = T.tensor(observation, dtype=T.float).to(self.device)
        return self.critic_network(obs)

class Agent:
    def __init__(self, lr_dynamics, lr_inner, lr_outer, env, n_models, 
                    fc1_dims, fc2_dims, n_steps):
        self.obs_dims = env.observation_space.shape
        self.act_dims = env.action_space.n
        self.dynamics = DynamicsModel(lr_dynamics, *self.obs_dims, len([self.act_dims]), 
                                      fc1_dims, fc2_dims)
        self.actor = ActorNetwork(lr_outer, *self.obs_dims, fc1_dims, \
                                    fc2_dims, self.act_dims)
        self.critic = CriticNetwork(lr_outer, *self.obs_dims, fc1_dims, fc2_dims)
        self.actor_inner = ActorNetwork(lr_inner, *self.obs_dims, fc1_dims, \
                                        fc2_dims, self.act_dims)
        self.transition_bank = Memory(n_steps)
        self.env = env
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.eta = 0.20
        self.n_updates = 2
        self.scores = []

    def choose_action(self, observation, real_env=False):
        if real_env:
            mu = Categorical(self.actor.forward(observation))
            action = mu.sample()
            log_probs = mu.log_prob(action)
            self.transition_bank.remember_log_probs(log_probs)
            return action.item()

        mu = Categorical(self.actor.forward(observation))
        action = mu.sample()
        log_probs = mu.log_prob(action)
        self.transition_bank.remember_log_probs(log_probs)
        return action.item()

    def sample_real_env(self):
        obs = self.env.reset()
        score = 0
        for _ in range(self.transition_bank.n_steps):
            act = self.choose_action(obs, real_env=True)
            obs_, rew, done, _ = self.env.step(act)
            self.transition_bank.remember_real(obs, act, rew, obs_, int(done))
            obs = obs_
            score += rew
            if done:
                obs = self.env.reset()
                self.scores.append(score)
                score = 0
        return np.mean(self.scores[-100:])


    def train_dynamics(self):
        obs, act, rew, obs_, dones = self.transition_bank.get_real_data()

        dynamics_loss = nn.MSELoss()

        self.dynamics.optimizer.zero_grad()
        new_obs_, new_rew, new_dones = self.dynamics.forward(obs, act)
        total_loss = dynamics_loss(new_obs_, obs_) + \
                        dynamics_loss(new_rew, rew) + \
                        dynamics_loss(new_dones, dones)
        total_loss.backward()
        self.dynamics.optimizer.step()

    def sample_imaginary_env(self, after_update=False):
        obs = self.env.reset()
        val = None
        for _ in range(self.transition_bank.n_steps):
            act = self.choose_action(obs)
            obs_, rew, done = self.dynamics.forward(obs, act)
            obs_ = obs_.detach().numpy()
            if after_update:
                val = self.critic.forward(obs)
            self.transition_bank.remember_imaginary(obs, act, rew, obs_, done, val)
            obs = obs_
            if done:
                obs = self.env.reset()

    def update_inner(self):
        rewards, log_probs = self.transition_bank.get_imag_data(inner=True)
        discounted_rewards = []
        for t in range(len(rewards)):
            disc_rewards = 0
            gamma = 1
            for reward in rewards[t:]:
                disc_rewards += reward * gamma
                gamma *= self.gamma
            discounted_rewards.append(disc_rewards)
        
        discounted_rewards = T.tensor(discounted_rewards, dtype=T.float).to(self.actor.device)
        log_probs = T.stack(log_probs).to(self.actor.device)
        
        discounted_rewards = (discounted_rewards - T.mean(discounted_rewards)) / T.std(discounted_rewards, dim=0)

        policy_loss = 0
        for return_estimate, log_prob in zip(discounted_rewards, log_probs):
            policy_loss -= return_estimate * log_prob
        
        self.actor_inner.optimizer.zero_grad()
        policy_loss.backward()
        self.actor_inner.optimizer.step()

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
        advantages = np.array(advantages, dtype=float)
        return advantages

    def update_outer(self):
        for _ in range(self.n_updates):
            observations, actions, rewards, _, dones, \
                log_probs, values, batches = self.transition_bank.get_imag_data()
            advantages = self.calc_advantages(rewards, values, dones)
            advantages = T.tensor(advantages, dtype=T.float).squeeze().to(self.actor.device)

            for batch in batches:
                obs = observations[batch]
                act = actions[batch]
                pi = self.actor.forward(obs)
                new_vals = self.critic.forward(obs)
                dist = Categorical(pi)
                old_probs = log_probs[batch]
                new_probs = dist.log_prob(act)

                probs_ratio = (new_probs - old_probs).exp()  ## (a/b) = log(a/b).exp() = (log(a) - log(b)).exp()
                clamped_ratio = probs_ratio.clamp(1 - self.eta, 1 + self.eta)

                #if T.sum((new_probs - old_probs) * new_probs.exp()) > self.early_stop:
                #    return
                actor_loss = -T.min(probs_ratio * advantages[batch], clamped_ratio * advantages[batch]).mean()
                critic_loss = T.mean((advantages[batch] + (values[batch] - new_vals.squeeze())).pow(2))
                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()


    def run(self):
        while mean < 175:
            mean = self.sample_real_env()
            self.train_dynamics()
            self.actor_inner.load_state_dict(self.actor.state_dict())
            self.sample_imaginary_env()
            self.update_inner()
            self.sample_imaginary_env(after_update=True)
            self.update_outer()
            print(f'Mean score: {np.mean(self.scores[-10:])}')



if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(lr_dynamics=1e-3, lr_inner=1e-3, lr_outer=1e-4, 
                    env=env, n_models=1, fc1_dims=256, fc2_dims=256,
                    n_steps=2500)
    agent.run()
