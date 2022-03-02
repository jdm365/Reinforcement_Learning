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
    def __init__(self, input_dims, n_actions):
        super(Connect4NetworkConvolutional, self).__init__()
        self.actor_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.Linear(input_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.Linear(input_dims, 1),
            nn.Tanh()
        )


    def block(self, in_filters=256):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_filters, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256)
        )
        return block

