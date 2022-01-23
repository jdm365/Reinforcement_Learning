from distutils.log import info
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as mp
from ActorCritic_Network import ActorCritic
from Memory import Memory
from Wrappers import make_env
from utils import plot_learning


def worker(name, input_shape, n_actions, global_agent, optimizer, 
    env_id, n_threads, global_idx):
    T_MAX = 20
    local_agent = ActorCritic(input_dims=input_shape, n_actions=n_actions)
    memory = Memory()
    frame_buffer = [input_shape[1], input_shape[2], 1]
    env = make_env(env_id, shape=frame_buffer)

    episode, max_eps, t_steps, scores = 0, 1000, 0, []
    while episode < max_eps:
        obs = env.reset()
        score, done, ep_steps = 0, False, 0
        hx = T.zeros(1, 256)
        while not done:
            state = T.tensor([obs], dtype=T.float)
            action, value, log_prob, hx = local_agent(state, hx)
            obs_, reward, done, info = env.step(action)
            memory.remember(reward, value, log_prob)
            score += reward
            obs = obs_
            ep_steps += 1
            t_steps += 1
            if ep_steps % T_MAX == 0 or done:
                rewards, values, log_probs = memory.sample_memory()
                loss = local_agent.calc_cost(obs, hx, done, rewards, values, log_probs)
                optimizer.zero_grad()
                hx = hx.detach_()
                loss.backward()
                T.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)
                for local_param, global_param in zip(local_agent.parameters(), global_agent.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                local_agent.load_state_dict(global_agent.state_dict())
                memory.clear_memory()
        episode += 1
        if name == '1':
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print(f'Episode {episode}', f'Average Score: {avg_score}', f'Thread {name}', f'Steps {t_steps}')

    if name == '1':
        x = [z for z in range(episode)]
        plot_learning(x, scores, 'A3C_Learning_Curve')