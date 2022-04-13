import gym
import numpy as np
from utils import plot_learning
from ddpg import Agent
import sys

def norm_action(action, env):
    high = env.action_space.high
    low = env.action_space.low
    return action - low / (high - low)


env = gym.make('LunarLanderContinuous-v2')
agent = Agent(fc1_dims=400, fc2_dims=300, lr_actor=3e-5, lr_critic=3e-4, \
    tau=1e-3, input_dims=env.observation_space.shape, \
    n_actions=env.action_space.shape[0])
n_episodes = 2500
filename = 'LunarLanderContinuous_DDPG'
best_score = env.reward_range[0]
score_history = []
render = False
np.random.seed(128)
for i in range(n_episodes):
    observation = env.reset()
    done = False
    score = 0
    j = 0
    agent.noise.reset()
    while not done:
        action = norm_action(agent.choose_action(observation), env)
        observation_, reward, done, _ = env.step(action)
        agent.store_memory(observation, action, reward, observation_, int(done))
        agent.learn()
        score += reward
        observation = observation_
        if render:
            env.render()
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()
    
    if avg_score > 160:
        render = True

    print(f'Episode {i} Score: {np.round(score, 2)} \t\t Average Score: {np.round(avg_score, 2)}')
x = [i+1 for i in range(n_episodes)]
plot_learning(score_history, filename=filename, x=x)

