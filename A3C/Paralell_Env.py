from ActorCritic_Network import ActorCritic
from Shared_Adam import SharedAdam
from Worker import worker
import torch.multiprocessing as mp

class ParellelEnv:
    def __init__(self, env_id, global_idx, input_shape, n_actions, num_threads):
        names = [str(i) for i in range(1, num_threads + 1)]
        
        global_actor_critic = ActorCritic(input_shape, n_actions)
        global_actor_critic.share_memory()
        global_optim = SharedAdam(global_actor_critic.parameters(), lr=1e-4)

        self.ps = [mp.Process(target=worker, args=(name, input_shape, n_actions, 
            global_actor_critic, global_optim, env_id, num_threads, global_idx)) 
            for name in names]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]