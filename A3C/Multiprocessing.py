import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import os
import torch.multiprocessing as mp

os.environ['OMP_NUM_THREADS'] = '1'

class ParellelEnv:
    def __init__(self, env_id, num_threads):
        names = [str(i) for i in range(num_threads)]
        
        self.ps = [mp.Process(target=worker, args=(name, env_id)) for name in names]
        [p.start() for p in self.ps]
        [p.join() for p in self.ps]

def worker(name, env_id):
    env = gym.make(env_id)
    episode, max_eps, scores = 0, 10, []
    while episode < max_eps:
        obs = env.reset()
        score, done = 0, False
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, _ = env.step(action)
            score += reward
            obs = obs_
        scores.append(score)
        print('episode {} process {} score {:.2f} '
            .format(episode, name, score))
        episode += 1

if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_id = 'CartPole-v0'
    n_threads = 4
    env = ParellelEnv(env_id=env_id, num_threads=n_threads)