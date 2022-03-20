import numpy as np
import random
import torch as T

class ReplayBuffer:
    def __init__(self, batch_size, n_actions,max_mem_length=750, unroll_length=5):
        self.batch_size = batch_size
        self.unroll_length = unroll_length
        self.n_actions = n_actions

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
        games = random.choices(self.games, k=self.batch_size)
        state_shape = games[0][0].shape

        states = np.zeros((self.batch_size, 1, *state_shape), dtype=T.float)
        probs = T.zeros((self.batch_size, self.unroll_length, self.n_actions), dtype=T.float)
        rewards = T.zeros((self.batch_size, self.unroll_length, 1), dtype=T.float)
        values = T.zeros((self.batch_size, self.unroll_length, 1), dtype=T.float)
        actions = T.zeros((self.batch_size, self.unroll_length, 1), dtype=T.float)

        for game_idx, game in enumerate(games):
            game_state, game_probs, game_rewards, game_values, game_actions = game
            idx = np.random.randint(0, len(game_state)-self.unroll_length, 1)[0]

            states[game_idx, 0] = game_state[idx]
            probs[game_idx, :, :] = T.tensor(game_probs[idx:idx+self.unroll_length], \
                dtype=T.float)
            rewards[game_idx, :, 0] = T.tensor(game_rewards[idx:idx+self.unroll_length], \
                dtype=T.float)
            values[game_idx, :, 0] = T.tensor(game_values[idx:idx+self.unroll_length], \
                dtype=T.float)
            actions[game_idx, :, 0] = T.tensor(game_actions[idx:idx+self.unroll_length], \
                dtype=T.long)
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
