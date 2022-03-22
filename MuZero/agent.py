import torch as T
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from networks_joined import Connect4NetworkConvolutional
from monte_carlo_tree_search import MCTS, Node
import pygame


class Agent:
    def __init__(self, lr, batch_size, n_simulations, hidden_state_dims, game, convolutional=True):
        self.network = Connect4NetworkConvolutional(
                            lr, 
                            game.input_dims, 
                            hidden_state_dims, 
                            game.n_actions
                            )
        self.memory = ReplayBuffer(batch_size, game.n_actions)
        self.tree_search = MCTS(self.network, n_simulations)
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
            hidden_state = self.network.project_to_hidden_state(state)
            self.tree_search.expand_node(root, self.game.get_valid_moves(state), \
                hidden_state, reward)
            root = self.tree_search.search(root)
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

        target_probs = target_probs.to(self.network.device)
        target_vals = target_vals.to(self.network.device)
        target_rewards = target_rewards.to(self.network.device)

        self.network.eval()

        initial_hidden_states = self.network.project_to_hidden_state(states).to(self.network.device)

        hidden_states = []
        rewards = []
        last_hidden_states = initial_hidden_states
        for i in range(self.memory.unroll_length):
            next_hidden_states, next_rewards = \
                self.network.roll_forward(last_hidden_states, target_actions[:, i, 0])
            hidden_states.append(next_hidden_states.to(self.network.device))
            rewards.append(next_rewards.squeeze().to(self.network.device))
            last_hidden_states = next_hidden_states

        probs = []
        vals = []
        for state in hidden_states:
            probabilities, values = self.network.actor_critic(state)
            probs.append(probabilities)
            vals.append(values.squeeze())

        self.network.train()
       
        actor_loss = 0
        critic_loss = 0
        reward_loss = 0
        for i in range(self.memory.unroll_length):
            actor_loss += -target_probs[:, i, :] * T.log(probs[i])
            critic_loss += F.mse_loss(target_vals[:, i, 0], vals[i])
            reward_loss += F.mse_loss(target_rewards[:, i, 0], rewards[i])
        total_loss = actor_loss.sum(dim=1).mean() + critic_loss + reward_loss

        self.network.optimizer.zero_grad()
        total_loss.backward()
        self.network.optimizer.step()

    def save_model(self):
        print('...Saving Models...')
        self.network.save_models()

    def load_model(self):
        print('...Loading Models...')
        self.network.load_models()

    def play_agent(self):
        self.load_model()
        game_over = False
        clicked = False
        while not game_over:
            temperature = 0.2

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
                root = Node(prior=0.1428)
                hidden_state = self.network.project_to_hidden_state(state)
                self.tree_search.expand_node(root, self.game.get_valid_moves(state), \
                    hidden_state, reward)
                root = self.tree_search.search(root)
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
