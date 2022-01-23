class Memory:
    def __init__(self):
        self.rewards = []
        self.values = []
        self.log_probs = []

    def remember(self, reward, value, log_prob):
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def clear_memory(self):
        self.rewards = []
        self.values = []
        self.log_probs = []
    
    def sample_memory(self):
        return self.rewards, self.values, self.log_probs