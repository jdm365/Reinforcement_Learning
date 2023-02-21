import time
import torch as T
from torch.utils.data import DataLoader
from replay_buffer import ReplayBufferDataset
from networks import ActorCriticNetwork
from games import ConnectN, Connect4
from monte_carlo_tree_search import MCTS, Node
import pygame


class Agent:
    def __init__(self, lr, batch_size, n_simulations, game=Connect4(), convolutional=True):
        self.actor_critic = ActorCriticNetwork(
                lr=lr, 
                input_dims=game.init_state.shape, 
                n_actions=game.columns, 
                convolutional=convolutional
                )

        model = self.actor_critic.eval().to(self.actor_critic.device)
        self.tree_search = MCTS(model, n_simulations, game)
        self.memory = ReplayBufferDataset(batch_size)
        self.batch_size = batch_size
        self.game = game


    def backup(self, search_path, value):
        '''
        Backup the value of the leaf node to the root node.
        '''
        rewards = []
        for i in range(len(search_path)):
            factor = pow(-1, i+1)
            rewards.append(value * factor)
        return list(reversed(rewards))


    def play_game(self):
        '''
        Play a game and store the states, actions and rewards.
        '''
        ## Always start from root node.
        temperature = 1.0
        node = self.tree_search.run()
        action, probs = node.choose_action(temperature)
        #self.memory.remember(np.copy(node.state), probs)
        self.memory.remember(node.state, probs)
        ## New root node
        node = Node(
                prior=probs[action], 
                game=self.game,
                prev_state=node.state, 
                prev_action=action 
                )
        value = self.game.get_reward(node.state)
        game_idx = 1

        while value is None:
            node = self.tree_search.run(node)
            action, probs = node.choose_action(temperature)
            #self.memory.remember(np.copy(node.state), probs)
            self.memory.remember(node.state, probs)

            ## New root node
            node = Node(
                    prior=probs[action], 
                    game=self.game, 
                    prev_state=node.state, 
                    prev_action=action
                    )
            value = self.game.get_reward(node.state)
            game_idx += 1
            if game_idx == 15:
                temperature = 0.01
        
        #self.memory.remember(np.copy(node.state), probs)
        self.memory.remember(node.state, probs)
        self.memory.episode_rewards = self.backup(self.memory.episode_states, value)

        self.memory.store_episode()


    def learn(self, n_steps=100):
        '''
        Update the actor-critic network.
        '''
        dataloader = DataLoader(
                self.memory,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2
                )

        for idx, (states, target_probs, target_vals) in enumerate(dataloader):
            if idx == n_steps:
                break

            self.actor_critic.optimizer.zero_grad()

            target_probs = target_probs.to(self.actor_critic.device)
            target_vals = target_vals.to(self.actor_critic.device)
            #self.actor_critic.eval()
            probs, vals = self.actor_critic.forward(states)
            #self.actor_critic.train()

            actor_loss = -(target_probs * T.log(probs)).sum(dim=1)
            critic_loss = T.sum((target_vals - vals.view(-1))**2) / self.batch_size
            total_loss = actor_loss.mean() + critic_loss

            total_loss.backward()
            self.actor_critic.optimizer.step()


    def save_model(self):
        print('...Saving Models...')
        self.actor_critic.save_models()


    def load_model(self, cpu=False):
        print('...Loading Models...')
        self.actor_critic.load_models(cpu)


    def play_agent(self, cpu=False):
        '''
        Play against the trained agent with pygame interface.
        '''
        self.load_model(cpu)
        game_over = False
        clicked = False
        while not game_over:
            temperature = 0.0

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
            node = Node(
                    prior=0.1428, 
                    game=self.game, 
                    prev_state=initial_state, 
                    prev_action=action
                    )
            value = self.game.get_reward(node.state)

            while value is None:
                self.game.draw_board(-node.state)
                node = self.tree_search.run(node)
                action, probs = node.choose_action(temperature)
                ## New root node
                node = Node(
                        prior=probs[action], 
                        game=self.game,
                        prev_state=node.state, 
                        prev_action=action
                        )
                value = self.game.get_reward(node.state)
                if value is not None:
                    winner = 'You lost loser MWAHAHAHAH!' if value == -1 else 'You won, darn'
                    game_over = True
                    self.game.draw_board(node.state, winner=winner)
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
                node = Node(
                        prior=0.1428, 
                        game=self.game,
                        prev_state=node.state, 
                        prev_action=action
                        )
                value = self.game.get_reward(node.state)
                if value is not None:
                    winner = 'You lost loser MWAHAHAHAH!' if value == -1 else 'You won, darn'
                    game_over = True
                    self.game.draw_board(-node.state, winner=winner)
                    break
        time.sleep(3)
