import numpy as np
import torch as T
from replay_buffer import ReplayBuffer
from networks import ActorCriticNetwork, RepresentationNetwork, DynamicsNetwork
from games import Connect4
from monte_carlo_tree_search import MCTS, Node
import pygame


class Agent:
    def __init__(self, lr, batch_size, n_simulations, hidden_state_dims, game, convolutional=True):
        self.actor_critic = ActorCriticNetwork(lr, hidden_state_dims, game.n_actions, convolutional)
        self.representation = RepresentationNetwork(lr, game.input_dims, hidden_state_dims)
        self.dynamics = DynamicsNetwork(lr, hidden_state_dims)
        self.tree_search = MCTS(self.actor_critic, self.representation, self.dynamics, n_simulations, game)
        self.memory = ReplayBuffer(batch_size)
        self.batch_size = batch_size
        self.game = game

    def backup(self, search_path, value):
        rewards = []
        for i in range(len(search_path)):
            factor = pow(-1, i+1)
            rewards.append(value * factor)
        return list(reversed(rewards))

    def play_game(self):
        self.game.state = self.game.init_state
        while self.game.check_terminal is False:
            root = Node(prior=0)
            hidden_state = self.representation.forward(self.game.state)
            self.tree_search.expand_node(root, 1, self.game.get_valid_moves(), hidden_state)
            node = self.tree_search.search(root)
            action, probs = self.tree_search.choose_action(1.0, self.game.columns)
            self.memory.remember(self.game.state, probs)
            self.game.iter_state(action)
        self.backup(self.memory.episode_values, self.game.get_reward())
        self.memory.store_episode()

    def learn(self):
        states, target_probs, target_vals = self.memory.get_batch()
        





        
        self.actor_critic.eval()
        probs, vals = self.actor_critic.forward(states)
        self.actor_critic.train()

        actor_loss = -(target_probs * T.log(probs)).sum(dim=1)
        critic_loss = T.sum((target_vals - vals.view(-1))**2) / self.batch_size
        total_loss = actor_loss.mean() + critic_loss

        self.actor_critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor_critic.optimizer.step()

    def save_model(self):
        print('...Saving Models...')
        self.actor_critic.save_models()

    def load_model(self, cpu=False):
        print('...Loading Models...')
        self.actor_critic.load_models(cpu)


    def play_agent(self, cpu=False):
        self.load_model(cpu)
        game_over = False
        clicked = False
        while not game_over:
            temperature = 0

            initial_state = self.game.init_state
            self.game.draw_board(initial_state)
            while not clicked:
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        posx = event.pos[0]
                        clicked = True
            clicked = False
            action = posx // 100
            ## New root node
            node = Node(prior=0.1428, prev_state=initial_state, prev_action=action, game=self.game)
            value = self.game.get_reward(node.state)

            while value is None:
                node = self.tree_search.run(node)
                action, probs = node.choose_action(temperature)
                ## New root node
                node = Node(prior=probs[action], prev_state=node.state, prev_action=action, game=self.game)
                value = self.game.get_reward(node.state)
                if value is not None:
                    winner = 'You lost loser MWAHAHAHAH!'
                    break
                self.game.draw_board(node.state)
                while not clicked:
                    for event in pygame.event.get():
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            posx = event.pos[0]
                            clicked = True
                clicked = False
                action = posx // 100
                ## New root node
                node = Node(prior=0.1428, prev_state=node.state, prev_action=action, game=self.game)
                value = self.game.get_reward(node.state)
                winner = 'You won, darn'
            game_over = True
            self.game.draw_board(node.state, winner=winner)
