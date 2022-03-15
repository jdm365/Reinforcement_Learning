from tqdm import tqdm
from agent import Agent
from games import Connect4


if __name__ == '__main__':
    agent = Agent(lr=1e-2, batch_size=2, n_simulations=10, \
        hidden_state_dims=(6, 6, 7), game=Connect4(), convolutional=True)

    def train(n_epochs):
        for epoch in tqdm(range(n_epochs)):
            agent.play_game()
            if len(agent.memory.games) > 2:
                agent.learn()

            if epoch % (n_epochs / 5) == 0:
                agent.save_model()
        agent.save_model()

    def test(cpu=False):
        agent.play_agent(cpu)


    n_epochs = 1000
    train(n_epochs=n_epochs)
    ##test()