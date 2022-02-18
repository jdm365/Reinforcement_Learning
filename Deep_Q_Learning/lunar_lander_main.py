import gym
import numpy as np
from utils import plot_learning
from lunar_lander import Agent

env = gym.make('LunarLander-v2')
num_episodes = 3000
agent = Agent(lr=5e-6, input_dims=[8], n_actions=4, fc1_dims=2058, fc2_dims=1536, gamma=0.99)

filename = 'Actor_critic_lunar_lander.png'

scores = []
for i in range(num_episodes):
    done = False
    observation = env.reset()
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        score += reward
        agent.learn(observation, reward, observation_, done)
        observation = observation_

        if i % 100 > 5 and i % 100 < 10:
            env.render()
    scores.append(score)

    avg_score = np.mean(scores[-100:])
    print('Episode ', i, 'score %.1f' % score, \
        'average score %.1f' % avg_score)
x = [i+1 for i in range(num_episodes)]
plot_learning(scores, filename=filename, x=x)