import gym
import numpy as np
from utils import plot_learning
from td3 import Agent
import sys

env = gym.make('BipedalWalker-v3')
agent = Agent(fc1_dims=400, fc2_dims=300, lr_actor=1e-3, lr_critics=1e-3, \
    tau=5e-3, input_dims=env.observation_space.shape, \
    n_actions=env.action_space.shape[0], env=env)
n_episodes = 750
filename = 'BipedalWalker_TD3'
best_score = env.reward_range[0]
score_history = []
render = False
np.random.seed(128)
for i in range(n_episodes):
    observation = env.reset()
    done = False
    score = 0
    max_ep_length = 1000
    cntr = 0
    agent.noise.reset()
    q_val = 0
    while not done:
        action, q_val_ = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        td = reward + agent.gamma * q_val_ - q_val
        agent.store_memory(observation, action, reward, observation_, int(done), td)
        agent.learn()
        score += reward
        observation = observation_
        q_val = q_val_
        if render:
            env.render()
        cntr += 1
        if cntr == max_ep_length:
            done = True
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