import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from saved_networks import Connect4NetworkConvolutional


class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.filename = 'Trained_Models/actor_critic'
        self.input_dims = input_dims
        self.network = Connect4NetworkConvolutional(input_dims, n_actions)

        self.conv_block_1 = self.network.block(in_filters=1)
        self.conv_block_2 = self.network.block()
        self.conv_block_3 = self.network.block()
        self.conv_block_4 = self.network.block()

        self.actor_head = self.network.actor_head
        self.critic_head = self.network.critic_head

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        out = self.conv_block_1(state)
        state_ = self.network.connect_residual(state, out)

        out = self.conv_block_2(out)
        state_ = self.network.connect_residual(state_, out)

        out = self.conv_block_3(out)
        state_ = self.network.connect_residual(state_, out)

        out = self.conv_block_4(out)
        state_ = self.network.connect_residual(state_, out)

        probs = self.actor_head(state_)[0]
        value = self.critic_head(state_)[0]
        return probs, value

    def save_models(self):
        T.save(self.state_dict(), self.filename)

    def load_models(self, cpu=False):
        if cpu:
            self.load_state_dict(T.load(self.filename, map_location=T.device('cpu')))
            return
        self.load_state_dict(T.load(self.filename))


class RepresentationNetwork(nn.Module):
    def __init__(self, lr, input_dims, output_dims):
        super(RepresentationNetwork, self).__init__()
        self.filename = 'Trained_Models/representation'
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.network = Connect4NetworkConvolutional(input_dims)

        self.conv_block_1 = self.network.block(in_filters=input_dims[1])
        self.conv_block_2 = self.network.block()
        self.conv_block_3 = self.network.block()
        self.conv_block_4 = self.network.block()

        self.representation = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=output_dims[1], kernel_size=1),
            nn.BatchNorm2d(output_dims[1]),
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.Linear(input_dims[-2]*input_dims[-1], output_dims[-2]*output_dims[-1])
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        out = self.conv_block_1(state)
        state_ = self.network.connect_residual(state, out)

        out = self.conv_block_2(out)
        state_ = self.network.connect_residual(state_, out)

        out = self.conv_block_3(out)
        state_ = self.network.connect_residual(state_, out)

        out = self.conv_block_4(out)
        state_ = self.network.connect_residual(state_, out)

        representation_hidden_state = self.representation(state_).reshape(self.input_dims[0], *self.output_dims)
        return representation_hidden_state

    def save_models(self):
        T.save(self.state_dict(), self.filename)

    def load_models(self, cpu=False):
        if cpu:
            self.load_state_dict(T.load(self.filename, map_location=T.device('cpu')))
            return
        self.load_state_dict(T.load(self.filename))
        

class DynamicsNetwork(nn.Module):
    def __init__(self, lr, input_dims):
        super(DynamicsNetwork, self).__init__()
        self.filename = 'Trained_Models/dynamics'
        self.input_dims = input_dims
        self.network = Connect4NetworkConvolutional(input_dims)

        self.conv_block_1 = self.network.block(in_filters=input_dims[1])
        self.conv_block_2 = self.network.block()
        self.conv_block_3 = self.network.block()
        self.conv_block_4 = self.network.block()

        self.hidden_state_prediction_network = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=input_dims[1], kernel_size=1),
            nn.BatchNorm2d(input_dims[1])
        )
        self.reward_prediction_network = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(input_dims[-2]*input_dims[-1], 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        input = T.cat((state, action), dim=1).to(self.device)

        out = self.conv_block_1(input)
        state_ = self.network.connect_residual(input, out)

        out = self.conv_block_2(out)
        state_ = self.network.connect_residual(state_, out)

        out = self.conv_block_3(out)
        state_ = self.network.connect_residual(state_, out)

        out = self.conv_block_4(out)
        state_ = self.network.connect_residual(state_, out)

        predicted_hidden_state = self.hidden_state_prediction_network(state_)
        predicted_reward = self.reward_prediction_network(state_).squeeze()
        return predicted_hidden_state, predicted_reward

    def save_models(self):
        T.save(self.state_dict(), self.filename)

    def load_models(self, cpu=False):
        if cpu:
            self.load_state_dict(T.load(self.filename, map_location=T.device('cpu')))
            return
        self.load_state_dict(T.load(self.filename))