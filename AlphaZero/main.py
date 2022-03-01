from unittest import result
import numpy as np
from tqdm import tqdm
from agent import Agent


if __name__ == '__main__':
    agent = Agent(lr=1e-2, batch_size=64, fc1_dims=32, fc2_dims=32, n_simulations=20)
    n_epochs = 5
    learn_frequency = 1000
    learning_steps_per_epoch = 100        
    test = False

    for epoch in tqdm(range(n_epochs)):
        for game in range(learn_frequency):
            winner = agent.play_game(test)
        for _ in range(learning_steps_per_epoch):
            agent.learn()
        if epoch % (n_epochs / 5) == 0:
            agent.save_model()
        if epoch >= .6 * n_epochs:
            test = True
        



