import numpy as np
from tqdm import tqdm
from agent import Agent


if __name__ == '__main__':
    agent = Agent(lr=1e-4, batch_size=64, fc1_dims=32, fc2_dims=32, n_simulations=5)
    n_epochs = 100
    learn_frequency = 100
    learning_steps_per_batch = 2

    display = False
    for epoch in tqdm(range(n_epochs)):
        if epoch > .95 * n_epochs:
            display = True
        for game in range(learn_frequency):
            agent.play_game(display)
        for _ in range(learning_steps_per_batch):
            agent.learn()
        if epoch % (n_epochs / 5) == 0:
            agent.memory.clear_memory()
            agent.save_model()



