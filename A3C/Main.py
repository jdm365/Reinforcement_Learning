import os
import torch.multiprocessing as mp
from Paralell_Env import ParellelEnv

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_id = 'PongNoFrameskip-v4'
    global_ep = mp.Value('i', 0)
    n_threads = 16
    n_actions = 6
    input_shape = [4, 42, 42]
    env = ParellelEnv(env_id=env_id, num_threads=n_threads, n_actions=n_actions, 
        global_idx=global_ep, input_shape=input_shape)
