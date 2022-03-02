import torch as T
import torch.nn as nn
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.filename = 'Trained_Models/actor_critic'
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

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, board):
        state = T.FloatTensor(board).to(self.device)
        if len(state.shape) != 1:
            state = state.reshape(state.shape[0], 1, state.shape[-1])
        else:
            state = state.reshape(1, 1, state.shape[-1])
        return self.actor_head(state)[0][0], self.critic_head(state)[0][0]

    def save_models(self):
        T.save(self.state_dict(), self.filename)

    def load_models(self):
        T.load_state_dict(T.load(self.filename))