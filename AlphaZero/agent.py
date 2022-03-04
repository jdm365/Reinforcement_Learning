import numpy as np
import torch as T
from replay_buffer import ReplayBuffer
from networks import ActorCriticNetwork
from games import ConnectN, Connect4
from monte_carlo_tree_search import MCTS, Node


class Agent:
    def __init__(self, lr, batch_size, n_simulations, game=Connect4()):
        self.actor_critic = ActorCriticNetwork(lr, game.init_state.shape, game.columns)
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

    def play_game(self, test=False):
        ## Always start from root node.
        temperature = 1.0 * (1-int(test))
        node = self.tree_search.run()
        action, probs = node.choose_action(temperature)
        self.memory.remember(np.copy(node.state), probs)
        ## New root node
        node = Node(prior=probs[action], prev_state=node.state, prev_action=action, game=self.game)
        value = self.game.get_reward(node.state)

        while value is None:
            node = self.tree_search.run(node)
            action, probs = node.choose_action(temperature)
            self.memory.remember(np.copy(node.state), probs)
            ## New root node
            node = Node(prior=probs[action], prev_state=node.state, prev_action=action, game=self.game)
            value = self.game.get_reward(node.state)
        
        self.memory.remember(np.copy(node.state), probs)
        self.memory.episode_rewards = self.backup(self.memory.episode_states, value)

        winner = 'Player 1' if self.memory.episode_rewards[0] == 1 else 'Player 2'
        if self.memory.episode_rewards[0] == 0:
            winner = 'Nobody -> Draw'

        if test:
            print(self.memory.episode_states, '\nWinner is', winner, '\n')
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

    def load_model(self):
        print('...Loading Models...')
        self.actor_critic.load_models()


    def play_agent(self):
        self.load_model()
        temperature = 0

        initial_state = self.game.init_state
        print(f'{initial_state}')
        action = int(input('Enter move (0-6): '))
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
                winner = 'You lost loser hahahahaha'
                break
            print(f'{node.state}')
            action = int(input('Enter move (0-6):'))
            ## New root node
            node = Node(prior=0.1428, prev_state=node.state, prev_action=action, game=self.game)
            value = self.game.get_reward(node.state)
            winner = 'You won, darn'
        print(node.state)
        print(winner)