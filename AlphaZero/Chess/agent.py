import numpy as np
import torch as T
from replay_buffer import ReplayBuffer
from networks import ActorCriticNetwork
from games import Chess
from monte_carlo_tree_search import MCTS, Node
import pygame


class Agent:
    def __init__(self, lr, batch_size, n_simulations, game=Chess(), convolutional=True):
        self.actor_critic = ActorCriticNetwork(lr, (6, 8, 8), 8*8*82, convolutional)
        self.tree_search = MCTS(self.actor_critic, n_simulations, game)
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
        node = Node(prior=0, prev_board=node.board, prev_action=action, game=self.game)
        value = self.game.get_reward(node.board)
        game_idx = 1

        while value is None:
            node = self.tree_search.run(node)
            action, probs = node.choose_action(temperature)
            self.memory.remember(np.copy(node.state), probs)
            ## New root node
            node = Node(prior=0, prev_board=node.board, prev_action=action, game=self.game)
            value = self.game.get_reward(node.board)
            game_idx += 1
            if game_idx == 80:
                temperature = 0.01
            if game_idx % 2:
                print(f'Move number: {game_idx//2}')
        
        self.memory.remember(np.copy(node.state), probs)
        self.memory.episode_rewards = self.backup(self.memory.episode_states, value)

        self.memory.store_episode()

    def learn(self):
        scaler = T.cuda.amp.GradScaler()

        states, target_probs, target_vals = self.memory.get_batch()
        target_probs = target_probs.to(self.actor_critic.device)
        target_vals = target_vals.to(self.actor_critic.device)
        self.actor_critic.eval()
        with T.cuda.amp.autocast():
            probs, vals = self.actor_critic.forward(states)
            self.actor_critic.train()

            actor_loss = -(target_probs * T.log(probs)).sum(dim=1)
            critic_loss = T.sum((target_vals - vals.view(-1))**2) / self.batch_size
            total_loss = actor_loss.mean() + critic_loss

        self.actor_critic.optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(self.actor_critic.optimizer)
        scaler.update()
        
        #self.actor_critic.optimizer.step()

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
