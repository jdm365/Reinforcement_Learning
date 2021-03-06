from tqdm import tqdm
from agent import Agent


if __name__ == '__main__':
    agent = Agent(lr=1e-2, batch_size=256, n_simulations=75, convolutional=False)

    def train(n_epochs):
        for epoch in tqdm(range(n_epochs)):
            agent.play_game()
            if len(agent.memory.states) > agent.batch_size:
                agent.learn()

            if epoch % (n_epochs / 5) == 0:
                agent.save_model()
        agent.save_model()

    def test(cpu=False):
        agent.play_agent(cpu)


    n_epochs = 1000
    train(n_epochs=n_epochs)
    ##test()