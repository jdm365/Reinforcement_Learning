import numpy as np
import torch as T

class ReplayBuffer:
    def __init__(self, batch_size, max_mem_length=750, unroll_length=5):
        self.batch_size = batch_size
        self.unroll_length = unroll_length

        self.games = []
        self.episode_states = []
        self.episode_actions = []
        self.episode_values = []
        
        self.max_length = max_mem_length
 
    def remember(self, state, action_probs):
        self.episode_states.append(state)
        self.episode_action_probs.append(action_probs)

    def get_batch(self):
        index = np.random.randint(0, len(self.games), self.batch_size)
        games = self.games[index]

        states = []
        probs = []
        values = []

        for game in games:
            idx = np.random.randint(0, len(game)-self.unroll_length, 1)
            game_states, game_probs, game_values = game
            states.append([game_states[idx]])
            probs.append([game_probs[idx:idx+self.unroll_length]])
            values.append([game_values[idx:idx:self.unroll_length]])

        return states, probs, values

    def store_episode(self):
        if len(self.games) > self.max_length:
            self.games.pop(0)
        self.games += (self.episode_states, self.episode_action_probs, self.episode_values)

        self.episode_states = []
        self.episode_action_probs = []
        self.episode_values = []
