from unittest import result
import numpy as np
from tqdm import tqdm
from agent import Agent


if __name__ == '__main__':
    agent = Agent(lr=1e-2, batch_size=64, fc1_dims=32, fc2_dims=32, n_simulations=100)
    n_epochs = 500
    learn_frequency = 100
    learning_steps_per_batch = 25

    for epoch in tqdm(range(n_epochs)):
        for game in range(learn_frequency):
            winner = agent.play_game()
        for _ in range(learning_steps_per_batch):
            agent.learn()
        if epoch % (n_epochs / 5) == 0:
            agent.memory.clear_memory()
            agent.save_model()



