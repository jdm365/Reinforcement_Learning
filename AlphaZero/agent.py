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
        self.game = game

    def backprop(self, search_path, value):
        rewards = []
        for idx, state in enumerate(reversed(search_path)):
            rewards.append(value * (-1 ** (idx+1)))
        return reversed(rewards)

    def play_game(self, display_games=False):
        node = self.tree_search.run()
        action, probs = node.choose_action(temperature=0)
        self.memory.remember(node.state, probs, 1)
        node = Node(prior=probs[action], prev_state=node.state, prev_action=action, game=self.game)

        while node.game.check_terminal(node.state) is False:
            node = self.tree_search.run(node)
            action, probs = node.choose_action(temperature=0.1)
            self.memory.remember(node.state, probs, 1)
            node = Node(prior=probs[action], prev_state=node.state, prev_action=action, game=self.game)
        value = node.game.check_terminal(node.state)
        self.memory.episode_rewards = self.backprop(self.memory.episode_states, value)
        if display_games:
            print(self.memory.episode_states)
        self.memory.store_episode()

    def learn(self):
        states, target_probs, target_vals = self.memory.get_batch()
        target_probs = target_probs.to(self.actor_critic.device)
        target_vals = target_vals.to(self.actor_critic.device)
        probs, vals = self.actor_critic.forward(states)

        actor_loss = -(target_probs * T.log(probs)).sum(dim=1)
        critic_loss = T.sum((target_vals - vals.view(-1))**2) / self.batch_size
        total_loss = actor_loss.mean() + critic_loss.mean()

        self.actor_critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor_critic.optimizer.step()

    def save_model(self):
        print('...Saving Models...')
        self.actor_critic.save_models()

    def load_model(self):
        print('...Loading Models')
        self.actor_critic.load_models()