import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, max_len, batch_size, prioritization=0.75):
        self.max_len = max_len
        self.batch_size = batch_size
        self.alpha = prioritization

        self.states = []
        self.actions = []
        self.rewards = []
        self.states_ = []
        self.dones = []

        self.tds = []

    def store_memory(self, state, action, reward, state_, done, td):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states_.append(state_)
        self.dones.append(1-done)

        self.tds.append(td.item())

        if len(self.states) == self.max_len:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.states_.pop(0)
            self.dones.pop(0)
            self.tds.pop(0)

    def get_memory_batch(self):
        tds = np.array(self.tds, dtype=np.float32)
        tds = np.power(np.absolute(tds) + 1e-6, self.alpha)
        tds = tds / np.sum(tds)

        batch = np.random.choice(len(self.states), self.batch_size, p=tds)

        state_batch = np.array(self.states)[batch]
        action_batch = np.array(self.actions)[batch]
        reward_batch = np.array(self.rewards)[batch]
        _state_batch = np.array(self.states_)[batch]
        done_batch = np.array(self.dones, dtype=np.float32)[batch]
        return state_batch, action_batch, reward_batch, _state_batch, done_batch