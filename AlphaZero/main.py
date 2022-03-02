from tqdm import tqdm
from agent import Agent


if __name__ == '__main__':
    agent = Agent(lr=1e-3, batch_size=64, n_simulations=75)
    n_epochs = 1000
    test = False

    for epoch in tqdm(range(n_epochs)):
        agent.play_game(test)
        if len(agent.memory.states) > agent.batch_size:
            agent.learn()

        if epoch % (n_epochs / 5) == 0:
            agent.save_model()
        if epoch >= .95 * n_epochs:
            test = True
        



