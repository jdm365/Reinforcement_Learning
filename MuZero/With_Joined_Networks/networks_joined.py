import sys
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


class ResidualBlock(nn.Module):
    def __init__(self, in_featues, out_features, kernel_size, padding, stride=None):
        super(ResidualBlock, self).__init__()
        ## input_dims (batch_size, in_features, height, width)
        self.residual_connection = nn.Conv2d(in_channels=in_featues, out_channels=out_features, kernel_size=1)
        if in_featues == out_features:
            self.residual_connection = lambda x : x
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_featues, out_channels=out_features, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_features)
        )

    def forward(self, input):
        output = self.residual_connection(input) + self.block(input)
        return F.relu(output)

    
class Resnet(nn.Module):
    def __init__(self, input_dims, in_featues, n_residual_blocks, \
        output_features: list, kernel_sizes: list, paddings: list, \
        strides: list, is_classifier=False, n_classes=None):
        super(Resnet, self).__init__()
        output_features = deque(output_features)
        output_features.appendleft(in_featues)
        tower = [ResidualBlock(output_features[i], output_features[i+1], kernel_sizes[i], \
                paddings[i], strides[i]) for i in range(n_residual_blocks)]
        self.residual_tower_list = nn.ModuleList(tower)
        self.residual_tower = nn.Sequential(*self.residual_tower_list)
        self.is_classifier = is_classifier
        if is_classifier:
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(output_features[-1]*input_dims[-2]*input_dims[-1], n_classes)
            )

    def forward(self, input):
        residual_output = self.residual_tower(input)
        if self.is_classifier:
            residual_output = self.fc(residual_output)
        return residual_output



class Connect4NetworkConvolutional(nn.Module):
    def __init__(self, lr, input_dims, hidden_state_dims, n_actions=1, n_unroll_steps=5):
        super(Connect4NetworkConvolutional, self).__init__()
        self.filename = 'Trained_Models/connect4resnet'
        self.hidden_state_dims = hidden_state_dims
        self.n_actions = n_actions

        self.representation_head = Resnet(
                                    input_dims, 
                                    in_featues=1,
                                    n_residual_blocks=3,
                                    output_features=[128, 256, hidden_state_dims[0]],
                                    kernel_sizes=[3, 3, 3], 
                                    paddings=[1, 1, 1], 
                                    strides=[1, 1, 1]
                                    )

        self.dynamics_head = Resnet(
                                    input_dims, 
                                    in_featues=hidden_state_dims[0]+1,
                                    n_residual_blocks=3,
                                    output_features=[hidden_state_dims[0], 256, hidden_state_dims[0]],
                                    kernel_sizes=[3, 3, 3], 
                                    paddings=[1, 1, 1], 
                                    strides=[1, 1, 1]
                                    )
        self.reward_predicition = nn.Sequential(
                                    nn.Conv2d(in_channels=hidden_state_dims[0], out_channels=1, kernel_size=1),
                                    nn.BatchNorm2d(1),
                                    nn.Tanh(),
                                    nn.Flatten(start_dim=1),
                                    nn.Linear(input_dims[-2]*input_dims[-1], 1)
                                    )
        self.n_unroll_steps = n_unroll_steps

        self.actor_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_state_dims[0], out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.Flatten(start_dim=1),
            nn.Linear(hidden_state_dims[-2]*hidden_state_dims[-1], n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_state_dims[0], out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.Flatten(start_dim=1),
            nn.Linear(hidden_state_dims[-2]*hidden_state_dims[-1], 1),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def project_to_hidden_state(self, state):
        state = self.reshape_state(state)
        return self.representation_head(state)

    def roll_forward(self, hidden_state, action):
        action = self.broadcast_action(action)
        dynamics_input = T.cat((hidden_state, action), dim=1)
        next_hidden_state = self.dynamics_head(dynamics_input)
        reward = self.reward_predicition(next_hidden_state)
        return next_hidden_state, reward

    def actor_critic(self, hidden_state):
        probs = self.actor_head(hidden_state)
        vals = self.critic_head(hidden_state)
        return probs[0], vals

    def reshape_state(self, state):
        if len(state.shape) == 2:
            state = state.reshape(1, 1, *state.shape)
            return T.from_numpy(state).float().to(self.device)
        return T.from_numpy(state).float().to(self.device)

    def broadcast_action(self, action):
        if type(action) == int:
            action = F.one_hot(T.tensor(action), self.n_actions)
            action = action.repeat(1, 1,\
                self.hidden_state_dims[1], 1).reshape(1, 1, *self.hidden_state_dims[1:])
        else:
            action = F.one_hot(action.long(), self.n_actions)
            action = action.repeat(1, 1, self.hidden_state_dims[1], 1).reshape(action.shape[0], \
                1, *self.hidden_state_dims[1:])
        return action.to(self.device)

    def save_models(self):
        T.save(self.state_dict(), self.filename)

    def load_models(self):
        if T.cuda.is_available() != True:
            self.load_state_dict(T.load(self.filename, map_location=T.device('cpu')))
            return
        self.load_state_dict(T.load(self.filename))