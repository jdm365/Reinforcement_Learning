import gym
import numpy as np
from advantage_actor_critic import Agent
from utils import plot_learning

env = gym.make('LunarLander-v2')
agent = Agent(lr_actor=1e-4, lr_critic=3e-5, input_dims=env.observation_space.shape, \
    n_actions=env.action_space.n)
n_episodes = 5000
score_history = []
#agent.load_model()

for i in range(n_episodes):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        agent.learn(observation, reward, observation_, done)
        score += reward
        observation = observation_
        #env.render()
    score_history.append(score)
    print(f'Episode {i+1} score: {np.round(score, 2)} \t\t Average score: {np.round(np.mean(score_history[-100:]), 2)}')
x = [x for x in range(n_episodes)]
plot_learning(score_history, x=x, filename='A2C_lunar_lander.png')
agent.save_models()
