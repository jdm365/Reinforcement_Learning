import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from saved_networks import ConnectN1dNetwork, Connect4NetworkConvolutional

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.filename = 'Trained_Models/actor_critic'
        self.input_dims = input_dims
        network = Connect4NetworkConvolutional(input_dims, n_actions)

        self.conv_block_1 = network.block(in_filters=1)
        self.conv_block_2 = network.block()
        self.conv_block_3 = network.block()
        self.conv_block_4 = network.block()

        self.actor_head = network.actor_head
        self.critic_head = network.critic_head

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, board):
        state = self.prep_state(board)

        out = F.relu(self.conv_block_1(state) + state)
        out = F.relu(self.conv_block_2(out) + out)
        out = F.relu(self.conv_block_3(out) + out)
        out = F.relu(self.conv_block_4(out) + out)

        probs = self.actor_head(out)
        value = self.critic_head(out)
        return probs[0], value[0]

    def prep_state(self, state):
        state = T.FloatTensor(state).to(self.device)
        if len(state.shape) != len(self.input_dims):
            state = state.reshape(state.shape[0], 1, *state.shape[1:])
        else:
            state = state.reshape(1, 1, *state.shape)
        return state

    def save_models(self):
        T.save(self.state_dict(), self.filename)

    def load_models(self):
        T.load_state_dict(T.load(self.filename))