import numpy as np
#from torch2trt import torch2trt
import torch as T
from replay_buffer import ReplayBuffer
from networks import ActorCriticNetwork
from games import ConnectN, Connect4
from monte_carlo_tree_search import MCTS, Node
import pygame


class Agent:
    def __init__(self, lr, batch_size, n_simulations, game=Connect4(), convolutional=True):
        self.actor_critic = ActorCriticNetwork(lr, game.init_state.shape, game.columns, convolutional)

        model = self.actor_critic.eval().to(self.actor_critic.device)
        #example_input = T.ones(game.init_state.shape).to(self.actor_critic.device)
        #model_trt = torch2trt(model, [example_input])
        self.tree_search = MCTS(model, n_simulations, game)
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
        ## Always start from root node.
        temperature = 1.0
        node = self.tree_search.run()
        action, probs = node.choose_action(temperature)
        self.memory.remember(np.copy(node.state), probs)
        ## New root node
        node = Node(prior=probs[action], prev_state=node.state, prev_action=action, game=self.game)
        value = self.game.get_reward(node.state)
        game_idx = 1

        while value is None:
            node = self.tree_search.run(node)
            action, probs = node.choose_action(temperature)
            self.memory.remember(np.copy(node.state), probs)
            ## New root node
            node = Node(prior=probs[action], prev_state=node.state, prev_action=action, game=self.game)
            value = self.game.get_reward(node.state)
            game_idx += 1
            if game_idx == 15:
                temperature = 0.01
        
        self.memory.remember(np.copy(node.state), probs)
        self.memory.episode_rewards = self.backup(self.memory.episode_states, value)

        self.memory.store_episode()

    def learn(self):
        states, target_probs, target_vals = self.memory.get_batch()
        target_probs = target_probs.to(self.actor_critic.device)
        target_vals = target_vals.to(self.actor_critic.device)
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
            temperature = 0.2

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
