import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConnectN1dNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, n_actions):
        super(ConnectN1dNetwork, self).__init__()
        self.shared_network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )

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
        input_dims = input_dims[0] * input_dims[1]
        self.actor_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.Flatten(start_dim=1),
            nn.Linear(input_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.Flatten(start_dim=1),
            nn.Linear(input_dims, 1),
            nn.Tanh()
        )

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def block(self, in_filters=256):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_filters, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256)
        )
        return block

    def downsample(self, state, out_chan=256, stride=1):
        in_chan = state.shape[1]
        if in_chan == out_chan:
            return state
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_chan)
        )
        downsample = downsample.to(self.device)
        return downsample(state)

    def connect_residual(self, state, block_output, out_chan=256, stride=1):
        residual_connection = self.downsample(state, out_chan, stride)
        return F.relu(block_output + residual_connection)

