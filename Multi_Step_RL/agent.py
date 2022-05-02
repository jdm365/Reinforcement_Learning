import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import Transformer



class Preprocess(nn.Module):
    def __init__(self, input_dims, state_dims, action_dims, embedding_dims, lr):
        super(Preprocess, self).__init__()
        self.input_dims = input_dims
        self.state_dims = state_dims
        self.action_dims = action_dims

        self.proj = nn.Linear(state_dims + action_dims, embedding_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_action = T.cat((state, action), dim=-1)
        return F.relu(self.proj(state_action))


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, embedding_dims, lr):
        super(ActorCriticNetwork, self).__init__()
        self.transformer = Transformer(input_dims, embedding_dims, n_heads, output_dims, 
                                       lr=lr, n_blocks=2)
        self.critic_network = nn.Sequential(
                nn.Linear(embedding_dims, fc1_dims),
                nn.LeakyReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.LeakyReLU(),
                nn.Linear(fc2_dims, 1)
                )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available())
        self.to(self.device)


    def forward(self, states, actions):
        ## states dims (batch_size, input_dims, embedding_dims)
        ## actions dims (batch_size, input_dims, 1)
        ## actor_output dims (batch_size, input_dims, output_dims)
        ## critic_output dims (batch_size, input_dims, 1)
        inputs = T.cat((states, actions), dim=-1)
        actor_outputs = self.transformer.forward(inputs)
        critic_outputs = self.critic_network(actor_outputs)
        return actor_outputs, critic_outputs


class Agent:
    def __init__(self, proj_lr, actor_critic_lr, state_dims, action_dims, 
                 fc1_dims, fc2_dims, sequence_length, embedding_dims):
        self.preprocessor = Preprocessor(sequence_length, state_dims, action_dims,
                                         embedding_dims, proj_lr)
        self.actor_critic = ActorCriticNetwork(sequence_length, fc1_dims, fc2_dims, 
                                               action_dims, embedding_dims, lr)
        

