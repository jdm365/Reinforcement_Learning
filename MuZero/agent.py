import numpy as np
import torch as T
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from networks import ActorCriticNetwork, RepresentationNetwork, DynamicsNetwork
from monte_carlo_tree_search import MCTS, Node
import pygame


class Agent:
    def __init__(self, lr, batch_size, n_simulations, hidden_state_dims, game, convolutional=True):
        self.actor_critic = ActorCriticNetwork(lr, hidden_state_dims, \
            game.n_actions, hidden_state_dims, convolutional)
        self.representation = RepresentationNetwork(lr, game.input_dims, hidden_state_dims)
        self.dynamics = DynamicsNetwork(lr, hidden_state_dims, game.n_actions)
        self.tree_search = MCTS(self.actor_critic, self.representation, \
            self.dynamics, n_simulations)
        self.memory = ReplayBuffer(batch_size, game.n_actions)
        self.batch_size = batch_size
        self.game = game

    def backup(self, n_moves, value):
        values = []
        for i in range(n_moves):
            factor = pow(-1, i+1)
            values.append(value * factor)
        return list(reversed(values))

    def play_game(self):
        state = self.game.init_state
        reward = 0
        root = Node(prior=0)
        while self.game.check_terminal(state) is False:
            #root = Node(prior=0)
            hidden_state = self.representation.forward(state)
            self.tree_search.expand_node(root, self.game.get_valid_moves(state), \
                hidden_state, reward)
            self.tree_search.search(root)
            action, probs = self.tree_search.choose_action(1.0, self.game.n_actions, \
                root)
            state = self.game.get_next_state(state, action)
            reward = self.game.get_reward(state)
            self.memory.remember(state, probs, reward, action)
            root = root.children[action]

        self.memory.episode_values = self.backup(len(self.memory.episode_states), \
                                                self.game.get_reward(state))
        self.memory.store_episode()

    def learn(self):
        states, target_probs, target_rewards, target_vals, target_actions = self.memory.get_batch()
        states = np.stack(states)
        states = np.expand_dims(states, axis=1)

        target_probs = target_probs.to(self.actor_critic.device)
        target_vals = target_vals.to(self.actor_critic.device)
        target_rewards = target_rewards.to(self.actor_critic.device)

        self.representation.eval()
        initial_hidden_states = self.representation.forward(states).to(self.dynamics.device)
        self.representation.train()

        self.dynamics.eval()
        hidden_states = []
        rewards = []
        last_hidden_states = initial_hidden_states
        for i in range(self.memory.unroll_length):
            next_hidden_states, next_rewards = \
                self.dynamics.forward(last_hidden_states, target_actions[:, i, 0])
            hidden_states.append(next_hidden_states.to(self.actor_critic.device))
            rewards.append(next_rewards.to(self.actor_critic.device))
            last_hidden_states = next_hidden_states
        self.dynamics.train()

        self.actor_critic.eval()
        probs = []
        vals = []
        for state in hidden_states:
            probabilities, values = self.actor_critic.forward(state)
            probs.append(probabilities)
            vals.append(values.squeeze())
        self.actor_critic.train()
       
        actor_loss = 0
        critic_loss = 0
        reward_loss = 0
        for i in range(self.memory.unroll_length):
            actor_loss += -target_probs[:, i, :] * T.log(probs[i])
            critic_loss += F.mse_loss(target_vals[:, i, 0], vals[i])
            reward_loss += F.mse_loss(target_rewards[:, i, 0], rewards[i])
        total_loss = actor_loss.sum(dim=1).mean() + critic_loss + reward_loss

        self.actor_critic.optimizer.zero_grad()
        self.representation.optimizer.zero_grad()
        self.dynamics.optimizer.zero_grad()
        total_loss.backward()
        self.actor_critic.optimizer.step()
        self.representation.optimizer.step()
        self.dynamics.optimizer.step()

    def save_model(self):
        print('...Saving Models...')
        self.actor_critic.save_models()
        self.representation.save_models()
        self.dynamics.save_models()

    def load_model(self, cpu=False):
        print('...Loading Models...')
        self.actor_critic.load_models(cpu)
        self.representation.load_models(cpu)
        self.dynamics.load_models(cpu)


    def play_agent(self, cpu=False):
        self.load_model(cpu)
        game_over = False
        clicked = False
        while not game_over:
            temperature = 0

            state = self.game.init_state
            self.game.draw_board(state)
            while not clicked:
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        posx = event.pos[0]
                        clicked = True
            clicked = False
            action = posx // 100
            state = self.game.get_next_state(state, action)
            reward = self.game.get_reward(state)

            while self.game.check_terminal(state) is False:
                root = Node(prior=0)
                hidden_state = self.representation.forward(state)
                self.tree_search.expand_node(root, self.game.get_valid_moves(state), \
                    hidden_state, reward)
                self.tree_search.search(root)
                action, _ = self.tree_search.choose_action(temperature, self.game.n_actions, \
                    root)
                state = self.game.get_next_state(state, action)
                reward = self.game.get_reward(state)
                if self.game.check_terminal(state) != False:
                    winner = 'You lost loser MWAHAHAHAH!'
                    break
                self.game.draw_board(state)
                while not clicked:
                    for event in pygame.event.get():
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            posx = event.pos[0]
                            clicked = True
                clicked = False
                action = posx // 100
                state = self.game.get_next_state(state, action)
                reward = self.game.get_reward(state)
                winner = 'You won, darn'
            game_over = True
            self.game.draw_board(state, winner=winner)
