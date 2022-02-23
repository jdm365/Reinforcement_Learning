import gym
import numpy as np
from policy_gradient import Agent
from utils import plot_learning

env = gym.make('LunarLander-v2')
agent = Agent(lr=1e-4, input_dims=env.observation_space.shape, \
    n_actions=env.action_space.n)
n_episodes = 10000
score_history = []
#agent.load_model()

for i in range(n_episodes):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        agent.remember(reward)
        score += reward
        observation = observation_
        #env.render()
    score_history.append(score)
    agent.learn()
    print(f'Episode {i+1} score: {np.round(score, 2)} \t\t Average score: {np.round(np.mean(score_history[-100:]), 2)}')
x = [x for x in range(n_episodes)]
plot_learning(score_history, x=x, filename='REINFORCE_lunar_lander.png')
agent.save_model()
