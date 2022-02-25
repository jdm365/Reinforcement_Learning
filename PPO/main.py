import gym
import numpy as np
from utils import plot_learning
from proximal_policy_optimization import Agent

env_name = 'LunarLander-v2'

env = gym.make(env_name)
num_episodes = 3000
agent = Agent(lr=1e-4, input_dims=env.observation_space.shape, \
    fc1_dims=256, fc2_dims=256, n_actions=env.action_space.n)

filename = f'PPO_{env_name}.png'

scores = []
for i in range(num_episodes):
    done = False
    observation = env.reset()
    score = 0
    while not done:
        action, probs, value = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        agent.remember(observation, action, reward, value, probs, done)
        if agent.steps_taken % agent.horizon == 0:
            agent.learn()
        score += reward
        observation = observation_
        if np.mean(scores[-100:]) > 190:
            env.render()
    scores.append(score)

    avg_score = np.mean(scores[-100:])
    print('Episode ', i, 'score %.1f' % score, \
        'average score %.1f' % avg_score)
x = [i+1 for i in range(num_episodes)]
plot_learning(scores, filename=filename, x=x)