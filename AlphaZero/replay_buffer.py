import numpy as np
import torch as T


class ReplayBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.states = []
        self.action_probs = []
        self.rewards = []

        self.episode_states = []
        self.episode_action_probs = []
        self.episode_rewards = []
 
    def remember(self, state, action_probs, reward):
        self.episode_states.append(state)
        self.episode_action_probs.append(action_probs)
        self.episode_rewards.append(reward)

    def get_batch(self):
        index = np.random.randint(0, len(self.states), self.batch_size)

        states = np.array(self.states, dtype=float)[index]
        probs = np.array(self.action_probs, dtype=float)[index]
        rewards = np.array(self.rewards, dtype=float)[index]

        states = T.FloatTensor(states)
        probs = T.FloatTensor(probs)
        rewards = T.FloatTensor(rewards)

        return states, probs, rewards

    def clear_memory(self):
        self.states = []
        self.action_probs = []
        self.rewards = []

    def store_episode(self, reward):
        self.episode_rewards = [reward * R for R in self.episode_rewards]
        #print(self.episode_rewards)

        self.states += self.episode_states
        self.action_probs += self.episode_action_probs
        self.rewards += self.episode_rewards

        self.episode_states = []
        self.episode_action_probs = []
        self.episode_rewards = []