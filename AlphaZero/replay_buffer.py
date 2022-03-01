import numpy as np
import torch as T


class ReplayBuffer:
    def __init__(self, batch_size, max_mem_length=25000):
        self.batch_size = batch_size

        self.states = []
        self.action_probs = []
        self.rewards = []

        self.episode_states = []
        self.episode_action_probs = []
        self.episode_rewards = []
        
        self.max_length= max_mem_length
 
    def remember(self, state, action_probs):
        self.episode_states.append(state)
        self.episode_action_probs.append(action_probs)

    def get_batch(self):
        index = np.random.randint(0, len(self.states), self.batch_size)

        states = np.array(self.states, dtype=float)[index]
        probs = np.array(self.action_probs, dtype=float)[index]
        rewards = np.array(self.rewards, dtype=float)[index]

        states = T.FloatTensor(states)
        probs = T.FloatTensor(probs)
        rewards = T.FloatTensor(rewards)

        return states, probs, rewards

    def store_episode(self):
        self.states += self.episode_states
        self.action_probs += self.episode_action_probs
        self.rewards += self.episode_rewards

        self.states = self.states[-self.max_length:]
        self.action_probs = self.action_probs[-self.max_length:]
        self.rewards = self.rewards[-self.max_length:]

        self.episode_states = []
        self.episode_action_probs = []
        self.episode_rewards = []
