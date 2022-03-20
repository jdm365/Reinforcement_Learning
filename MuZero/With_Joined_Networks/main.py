from tqdm import tqdm
from agent import Agent
from games import Connect4


if __name__ == '__main__':
    agent = Agent(lr=1e-2, batch_size=64, n_simulations=200, \
        hidden_state_dims=(128, 6, 7), game=Connect4(), convolutional=True)

    def train(n_epochs):
        for epoch in tqdm(range(n_epochs)):
            agent.play_game()
            if len(agent.memory.games) > 10:
                agent.learn()

            if epoch % (n_epochs // 10) == 0:
                agent.save_model()
        agent.save_model()

    def test():
        agent.play_agent()


    n_epochs = 1000
    train(n_epochs=n_epochs)
    #test()