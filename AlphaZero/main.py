from tqdm import tqdm
from agent import Agent


if __name__ == '__main__':
    agent = Agent(lr=1e-2, batch_size=256, n_simulations=500)
    n_epochs = 1750

    def train(n_epochs):
        for epoch in tqdm(range(n_epochs)):
            agent.play_game()
            if len(agent.memory.states) > agent.batch_size:
                agent.learn()

            if epoch % (n_epochs / 5) == 0:
                agent.save_model()
        agent.save_model()

    def test():
        agent.play_agent()

    #train(n_epochs=n_epochs)
    test()