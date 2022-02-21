import gym
import numpy as np
from utils import plot_learning
from ddpg import Agent

env = gym.make('LunarLanderContinuous-v2')
agent = Agent(lr_actor=3e-5, lr_critic=3e-4, tau=1e-3, input_dims=env.observation_space.shape, \
    n_actions=env.action_space.shape[0])
n_episodes = 2500
filename = 'LunarLander_DDPG'
best_score = env.reward_range[0]
score_history = []
render = False

for i in range(n_episodes):
    observation = env.reset()
    done = False
    score = 0
    agent.noise.reset()
    while not done:
        action = agent.choose_action(observation)
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

