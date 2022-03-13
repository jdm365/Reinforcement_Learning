import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from saved_networks import ChessNetworkConvolutional, ChessNetworkTransformer

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, convolutional=True):
        super(ActorCriticNetwork, self).__init__()
        self.filename = 'Trained_Models/actor_critic'
        self.input_dims = input_dims
        self.convolutional = convolutional
        if convolutional:
            self.network = ChessNetworkConvolutional(input_dims, n_actions)

            self.conv_block_1 = self.network.block(in_filters=6)
            self.conv_block_2 = self.network.block()
            self.conv_block_3 = self.network.block()
            self.conv_block_4 = self.network.block()
            self.conv_block_5 = self.network.block()
            self.conv_block_6 = self.network.block()
            self.conv_block_7 = self.network.block()
            self.conv_block_8 = self.network.block()

            self.actor_head = self.network.actor_head
            self.critic_head = self.network.critic_head
        else:
            self.transformer = ChessNetworkTransformer(input_dims, n_actions, n_encoder_blocks=2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, board):
        state = self.prep_state(board)

        if self.convolutional:
            out = self.conv_block_1(state)
            state_ = self.network.connect_residual(state, out)

            out = self.conv_block_2(out)
            state_ = self.network.connect_residual(state_, out)

            out = self.conv_block_3(out)
            state_ = self.network.connect_residual(state_, out)

            out = self.conv_block_4(out)
            state_ = self.network.connect_residual(state_, out)

            out = self.conv_block_5(out)
            state_ = self.network.connect_residual(state_, out)

            out = self.conv_block_6(out)
            state_ = self.network.connect_residual(state_, out)

            out = self.conv_block_7(out)
            state_ = self.network.connect_residual(state_, out)

            out = self.conv_block_8(out)
            state_ = self.network.connect_residual(state_, out)

            probs = self.actor_head(state_)[0]
            value = self.critic_head(state_)[0]
        else:
            probs, value = self.transformer.forward(state)
            probs = probs[0]
            value = value[0]
        return probs, value

    def prep_state(self, state):
        state = T.FloatTensor(state).to(self.device)
        if len(state.shape) != len(self.input_dims):
            state = state.reshape(state.shape[0], *state.shape[1:])
        else:
            state = state.reshape(1, *state.shape)
        return state

    def save_models(self):
        T.save(self.state_dict(), self.filename)

    def load_models(self, cpu=False):
        if cpu:
            self.load_state_dict(T.load(self.filename, map_location=T.device('cpu')))
            return
        self.load_state_dict(T.load(self.filename))