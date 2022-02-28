import numpy as np
import torch as T
from replay_buffer import ReplayBuffer
from networks import ActorCriticNetwork
from games import ConnectN
from monte_carlo_tree_search import MCTS, Node


class Agent:
    def __init__(self, lr, batch_size, fc1_dims=32, fc2_dims=32, n_simulations=100, game=ConnectN()):
        self.actor_critic = ActorCriticNetwork(lr, game.columns, fc1_dims, fc2_dims, game.columns)
        self.tree_search = MCTS(self.actor_critic, n_simulations, game)
        self.memory = ReplayBuffer(batch_size)
        self.batch_size = batch_size

    def play_game(self):
        node = self.tree_search.run()
        action, probs = node.choose_action(temperature=0)
        self.memory.remember(node.state, probs, node.to_play)
        node = Node(prior=probs[action], to_play=-node.to_play, prev_state=node.state, prev_action=action)

        while node.game.check_terminal(node.state, node.to_play) is False:
            node = self.tree_search.run(node)
            action, probs = node.choose_action(temperature=0)
            self.memory.remember(node.state, probs, node.to_play)
            ## New root node; set prior prob to 0
            node = Node(prior=probs[action], to_play=-node.to_play, prev_state=node.state, prev_action=action)
        reward = node.game.check_terminal(node.state, node.to_play)
        self.memory.store_episode(reward)

    def learn(self):
        states, target_probs, target_vals = self.memory.get_batch()
        target_probs = target_probs.to(self.actor_critic.device)
        target_vals = target_vals.to(self.actor_critic.device)
        probs, vals = self.actor_critic.forward(states)

        actor_loss = -(target_probs * T.log(probs)).mean()
        critic_loss = T.mean((target_vals - vals.view(-1))**2) / self.batch_size
        total_loss = actor_loss + critic_loss

        self.actor_critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor_critic.optimizer.step()

    def save_model(self):
        print('...Saving Models...')
        self.actor_critic.save_models()

    def load_model(self):
        print('...Loading Models')
        self.actor_critic.load_models()