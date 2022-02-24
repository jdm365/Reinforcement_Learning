import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn

def plot_learning(scores, filename=None, x=None, window=100):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Profits')       
    plt.xlabel('Episode')                     
    plt.plot(x, running_avg)
    if filename:
      plt.savefig(filename)

def init_linear(network):
    for layer in network:
        if isinstance(layer, nn.Linear):
            f = 1./np.sqrt(layer.weight.data.shape[0])
            T.nn.init.uniform_(layer.weight, -f, f)
            T.nn.init.uniform_(layer.bias, -f, f)
