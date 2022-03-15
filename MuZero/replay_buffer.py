import numpy as np
import random
import torch as T

class ReplayBuffer:
    def __init__(self, batch_size, max_mem_length=750, unroll_length=5):
        self.batch_size = batch_size
        self.unroll_length = unroll_length

        self.games = []
        self.episode_states = []
        self.episode_action_probs = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_actions = []
        
        self.max_length = max_mem_length
 
    def remember(self, state, action_probs, reward, action):
        self.episode_states.append(state)
        self.episode_action_probs.append(action_probs)
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)

    def get_batch(self):
        games = random.sample(self.games, self.batch_size)

        states = []
        probs = []
        rewards = []
        values = []
        actions = []

        for game in games:
            game_state, game_probs, game_rewards, game_values, game_actions = game
            idx = np.random.randint(0, len(game_state)-self.unroll_length, 1)[0]

            states.append([game_state[idx]])
            probs.append([game_probs[idx:idx+self.unroll_length]])
            rewards.append([game_rewards[idx:idx:self.unroll_length]])
            values.append([game_values[idx:idx:self.unroll_length]])
            actions.append([game_actions[idx:idx:self.unroll_length]])

        return states, probs, rewards, values, actions

    def store_episode(self):
        if len(self.games) > self.max_length:
            self.games.pop(0)
        self.games.append((self.episode_states, self.episode_action_probs, \
            self.episode_rewards, self.episode_values, self.episode_actions))

        self.episode_states = []
        self.episode_action_probs = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_actions = []
