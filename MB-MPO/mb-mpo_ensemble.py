from fileinput import filename
import gym
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class RealMemory:
    def __init__(self, n_steps=100, batch_size=64, name=None):
        self.n_steps = n_steps
        self.batch_size = batch_size

        self.observations = []
        self.actions = []
        self.rewards = []
        self.observations_ = []
        self.dones = []

    def remember(self, obs, act, rew, obs_, done):
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.observations_.append(obs_)
        self.dones.append(done)

    def get_data(self):
        index = np.random.randint(0, len(self.dones), self.batch_size)

        obs = np.array(self.observations)[index]
        act = np.array(self.actions)[index]
        rew = T.FloatTensor(self.rewards)[index]
        obs_ = T.FloatTensor(self.observations_)[index]
        dones = T.FloatTensor(self.dones)[index]

        return obs, act, rew, obs_, dones

class ImaginaryMemory:
    def __init__(self, n_steps=100, batch_size=64, name=None):
        self.n_steps = n_steps
        self.n_steps_inner = 1000
        self.batch_size = batch_size

        self.observations = []
        self.actions = []
        self.rewards = []
        self.observations_ = []
        self.dones = []
        self.log_probs = []
        self.vals = []

    def remember(self, obs, act, rew, obs_, done, val):
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.observations_.append(obs_)
        self.dones.append(done)
        self.vals.append(val)

    def remember_log_probs(self, log_probs):
        self.log_probs.append(log_probs)

    def get_data_inner(self):
        rew = self.rewards
        log_probs = self.log_probs
        self.reset_memory()
        return rew, log_probs

    def get_data_outer(self):
        batch_start = np.arange(0, len(self.observations), self.batch_size)
        indices = np.arange(len(self.observations), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        obs = np.array(self.observations)
        act = T.FloatTensor(self.actions)
        rew = T.FloatTensor(self.rewards)
        obs_ = T.FloatTensor(self.observations_)
        dones = T.FloatTensor(self.dones)
        vals = T.FloatTensor(self.vals)
        log_probs = T.FloatTensor(self.log_probs)

        self.reset_memory()
        return obs, act, rew, obs_, dones, log_probs, vals, batches

    def reset_memory(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.observations_ = []
        self.dones = []
        self.log_probs = []
        self.vals = []


class DynamicsModel(nn.Module):
    def __init__(self, lr, obs_dims, act_dims, fc1_dims, fc2_dims, name=None):
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
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name=None):
        super(ActorNetwork, self).__init__()

        self.actor_network = nn.Sequential(
            nn.Linear(input_dims, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
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
        dynamics = [(str(i), DynamicsModel(lr_dynamics, *self.obs_dims, len([self.act_dims]), 
                                      fc1_dims, fc2_dims, name=str(i))) for i in range(n_models)]
        self.dynamics = dict(dynamics)
        self.critic_global = CriticNetwork(lr_outer, *self.obs_dims, fc1_dims, fc2_dims)
        actors = [(str(i), ActorNetwork(lr_inner, *self.obs_dims, fc1_dims, \
                                        fc2_dims, self.act_dims, name=str(i))) for i in range(n_models)]       
        actors.append(('Global', ActorNetwork(lr_outer, *self.obs_dims, fc1_dims, \
                                    fc2_dims, self.act_dims, name='Global')))              
        memories = [(str(i), ImaginaryMemory(n_steps, name=str(i))) for i in range(n_models)]
        memories.append(('Real', RealMemory(n_steps, name='Real')))

        self.actors = dict(actors)
        self.memories = dict(memories)
        self.names = [str(i) for i in range(n_models)]
        envs = [(str(i), env) for i in range(n_models)]
        self.envs = dict(envs)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.eta = 0.20
        self.n_updates = 2
        self.scores = []
        self.n_models = n_models

    def choose_action(self, observation, actor_name, memory_name):
        mu = Categorical(self.actors[actor_name].forward(observation))
        action = mu.sample()
        log_probs = mu.log_prob(action)
        if memory_name != 'Real':
            self.memories[memory_name].remember_log_probs(log_probs)
        return action.item()

    def sample_real_env(self, actor_name):
        obs = self.envs[actor_name].reset()
        score = 0
        for _ in range(self.memories['Real'].n_steps):
            act = self.choose_action(obs, actor_name, 'Real')
            obs_, rew, done, _ = self.envs[actor_name].step(act)
            self.memories['Real'].remember(obs, act, rew, obs_, int(done))
            obs = obs_
            score += rew
            if done:
                obs = self.envs[actor_name].reset()
                self.scores.append(score)
                score=0

    def train_dynamics(self, model_name):
        obs, act, rew, obs_, dones = self.memories['Real'].get_data()

        dynamics_loss = nn.MSELoss()

        self.dynamics[model_name].optimizer.zero_grad()
        new_obs_, new_rew, new_dones = self.dynamics[model_name].forward(obs, act)
        total_loss = dynamics_loss(new_obs_, obs_) + \
                        dynamics_loss(new_rew, rew) + \
                        dynamics_loss(new_dones, dones)
        total_loss.backward()
        self.dynamics[model_name].optimizer.step()

    def sample_imaginary_env(self, actor_name, model_name):
        obs = self.envs['0'].reset()
        val = None
        for _ in range(self.memories['0'].n_steps_inner):
            act = self.choose_action(obs, actor_name, actor_name)
            obs_, rew, done = self.dynamics[model_name].forward(obs, act)
            obs_ = obs_.detach().numpy()
            if actor_name != 'Global':
                val = self.critic_global.forward(obs)
            self.memories[actor_name].remember(obs, act, rew, obs_, done, val)
            obs = obs_
            if done:
                obs = self.envs['0'].reset()

    def update_inner(self, model_name):
        rewards, log_probs = self.memories[model_name].get_data_inner()
        discounted_rewards = []
        for t in range(len(rewards)):
            disc_rewards = 0
            gamma = 1
            for reward in rewards[t:]:
                disc_rewards += reward * gamma
                gamma *= self.gamma
            discounted_rewards.append(disc_rewards)
        
        discounted_rewards = T.tensor(discounted_rewards, dtype=T.float).to(self.actors[model_name].device)
        log_probs = T.stack(log_probs).to(self.actors[model_name].device)
        
        discounted_rewards = (discounted_rewards - T.mean(discounted_rewards)) / T.std(discounted_rewards, dim=0)

        policy_loss = 0
        for return_estimate, log_prob in zip(discounted_rewards, log_probs):
            policy_loss -= return_estimate * log_prob
        
        self.actors[model_name].optimizer.zero_grad()
        policy_loss.backward()
        self.actors[model_name].optimizer.step()

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

    def update_outer(self, actor_name):
        for _ in range(self.n_updates):
            observations, actions, rewards, _, dones, \
                log_probs, values, batches = self.memories[actor_name].get_data_outer()
            advantages = self.calc_advantages(rewards, values, dones)
            advantages = T.tensor(advantages, dtype=T.float).squeeze().to(self.actors['Global'].device)

            for batch in batches:
                obs = observations[batch]
                act = actions[batch]
                pi = self.actors['Global'].forward(obs)
                new_vals = self.critic_global.forward(obs)
                dist = Categorical(pi)
                old_probs = log_probs[batch]
                new_probs = dist.log_prob(act)

                probs_ratio = (new_probs - old_probs).exp()  ## (a/b) = log(a/b).exp() = (log(a) - log(b)).exp()
                clamped_ratio = probs_ratio.clamp(1 - self.eta, 1 + self.eta)

                #if T.sum((new_probs - old_probs) * new_probs.exp()) > self.early_stop:
                #    return
                actor_loss = -T.min(probs_ratio * advantages[batch], clamped_ratio * advantages[batch]).mean()
                critic_loss = T.mean((advantages[batch] + (values[batch] - new_vals.squeeze())).pow(2))
                total_loss = (actor_loss + 0.5 * critic_loss) / self.n_models

                self.actors['Global'].optimizer.zero_grad()
                self.critic_global.optimizer.zero_grad()
                total_loss.backward()
                self.actors['Global'].optimizer.step()
                self.critic_global.optimizer.step()


    def run(self):
        epoch = 0
        mean = 0
        while mean < 175:
            epoch += 1
            for _ in range(20):
                for name in self.names:
                    self.sample_real_env(name)
                    self.train_dynamics(name)
            for name in self.names:
                self.sample_imaginary_env(name, name)
                self.update_inner(name)
            for name in self.names:
                self.sample_imaginary_env(name, name)
                self.update_outer(name)
            mean = int(np.mean(self.scores[-25:]))
            print(f'Score running mean: {mean} \t Epoch: {epoch}')
        return self.actors['Global'].state_dict()



if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    filename = 'actor_params'
    agent = Agent(lr_dynamics=1e-3, lr_inner=1e-3, lr_outer=1e-4, 
                    env=env, n_models=8, fc1_dims=256, fc2_dims=256,
                    n_steps=1000)
    optimal_preupdate_params = agent.run()
    T.save(optimal_preupdate_params, 'Trained_Models/' + filename)
