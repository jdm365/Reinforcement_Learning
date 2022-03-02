import torch as T
import torch.nn as nn
import torch.optim as optim
from saved_networks import ConnectN1dNetwork, Connect4NetworkConvolutional

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.filename = 'Trained_Models/actor_critic'
        self.input_dims = input_dims
        network = Connect4NetworkConvolutional(input_dims, fc1_dims, fc2_dims, n_actions)

        self.actor_head = network.actor_head
        self.critic_head = network.critic_head

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, board):
        state = T.FloatTensor(board).to(self.device)
        if len(state.shape) != len(self.input_dims):
            state = state.reshape(state.shape[0], 1, *state.shape[1:])
        else:
            state = state.reshape(1, 1, *state.shape)
        return self.actor_head(state)[0][0][0], self.critic_head(state)[0][0][0]

    def save_models(self):
        T.save(self.state_dict(), self.filename)

    def load_models(self):
        T.load_state_dict(T.load(self.filename))