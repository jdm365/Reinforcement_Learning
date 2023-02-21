from tqdm import tqdm
from agent import Agent
#import torch.multiprocessing as mp

def train(agent, n_games, sim_expansion_factor=4):
    '''
    Train the agent.
    param agent: Agent object
    param n_games: Number of games to train for
    param sim_expansion_factor: Factor by which to increase the number of 
                                simulations for first 50 epochs. Hopefully
                                will provide more directed exploration in 
                                the early stages of training.
    '''
    agent.tree_search.n_simulations *= sim_expansion_factor

    for epoch in tqdm(range(n_games), desc='Training'):
        agent.play_game()

        if len(agent.memory.states) > agent.batch_size:
            ## Try multiple learning steps every few epochs to
            ## reduce device transfer latency.
            if epoch % 25 == 0:
                agent.learn(n_steps=100)


        if epoch == 50:
            agent.tree_search.n_simulations /= sim_expansion_factor
            agent.tree_search.n_simulations = int(agent.tree_search.n_simulations)

        if epoch % (n_games // 50) == 0:
            agent.save_model()

    agent.save_model()


def test(agent, cpu=False):
    agent.play_agent(cpu)






if __name__ == '__main__':
    agent = Agent(lr=1e-3, batch_size=4096, n_simulations=16, convolutional=False)

    n_games = 1000
    train(agent, n_games=n_games)
    #test(agent)
