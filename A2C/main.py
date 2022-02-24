import gym
import numpy as np
from advantage_actor_critic import Agent
from utils import plot_learning

env = gym.make('CartPole-v0')
agent = Agent(lr_actor=1e-5, lr_critic=3e-5, fc1_dims=512, fc2_dims=512, input_dims=env.observation_space.shape, \
    n_actions=env.action_space.n)
n_episodes = 3000
score_history = []
solved = False
agent.load_models()

for i in range(n_episodes):
    if solved:
        total_episodes = i
        break
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        agent.remember(observation, reward)
        if (len(agent.state_memory) == agent.max_mem_length+1) or done:
            agent.learn(observation_, done)
        score += reward
        observation = observation_
        #env.render()
    score_history.append(score)
    print(f'Episode {i+1} score: {np.round(score, 2)} \t\t Average score: {np.round(np.mean(score_history[-100:]), 2)}')
    if np.mean(score_history[-100:]) > 190:
        agent.save_models()
        solved = True
x = [x for x in range(total_episodes)]
plot_learning(score_history, x=x, filename='A2C_Cartpole.png')
