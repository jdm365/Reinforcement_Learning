import torch as T
import torch.nn as nn
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.filename = 'Trained_Models/actor_critic'
        self.actor_network = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic_network = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, 1),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, board):
        state = T.FloatTensor(board).to(self.device)
        return self.actor_network(state), self.critic_network(state)

    def save_models(self):
        T.save(self.state_dict(), self.filename)

    def load_models(self):
        T.load_state_dict(T.load(self.filename))