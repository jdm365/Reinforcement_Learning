import torch as T
import torch.nn as nn
import numpy as np


class ConnectN1dNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, n_actions):
        super(ConnectN1dNetwork, self).__init__()
        self.shared_network = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=32, out_channels=1, kernel_size=2),
                    nn.BatchNorm1d(1),
                    nn.ReLU()
                )
        input_dims = input_dims - 2

        self.actor_head = nn.Sequential(
            self.shared_network,
            nn.Linear(input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic_head = nn.Sequential(
            self.shared_network,
            nn.Linear(input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, 1),
            nn.Tanh()
        )

class Connect4NetworkConvolutional(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Connect4NetworkConvolutional, self).__init__()
        self.shared_network = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1024, kernel_size=(2,2)),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3)),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(3,3)),
                    nn.BatchNorm2d(1),
                    nn.ReLU()
                )

        post_conv_dims = T.tensor(input_dims, dtype=T.int)
        post_conv_dims[0] = input_dims[0] - 1 - 2 - 2
        post_conv_dims[1] = input_dims[1] - 1 - 2 - 2
        post_conv_dims = int(post_conv_dims[1])
        

        self.actor_head = nn.Sequential(
            self.shared_network,
            nn.Linear(post_conv_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic_head = nn.Sequential(
            self.shared_network,
            nn.Linear(post_conv_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, 1)
        )