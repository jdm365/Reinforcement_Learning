import enum
import numpy as np
from sympy import root
import torch as T
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

## 1d connect N game
class ConnectN:
    def __init__(self, N=2):
        self.columns = N + 2*(N // 2)
        self.N = N
        self.init_state = np.zeros(self.columns, dtype=int)

    def get_next_state(self, board, action, player):
        board_ = np.copy(board)
        board_[action] = player
        player *= -1
        return player * board_

    def get_valid_moves(self, board):
        valid_moves = np.zeros_like(board, dtype=int)
        moves_available = False
        for idx, space in enumerate(board):
            if space == 0:
                valid_moves[idx] = 1
                moves_available = True
        if moves_available:
            return valid_moves
        return False

    def check_terminal(self, board, player):
        for idx in range(self.columns-self.N):
            chunk = board[idx:idx+self.N]
            victory = len([None for x in chunk if x == player]) == self.N
            loss = len([None for x in chunk if x == -player]) == self.N
            if victory:
                return player
            if loss:
                return -player
        return False

    def get_player_reward(self, board, player):
        reward = self.check_terminal(board, player)
        if reward == False:
            return 0
        return reward
    
    def get_board(self, board, player):
        return player * board

class ReplayBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.states = []
        self.action_probs = []
        self.rewards = []

        self.episode_states = []
        self.episode_action_probs = []
        self.episode_rewards = []
 
    def remember(self, state, action_probs, reward):
        self.episode_states.append(state)
        self.episode_action_probs.append(action_probs)
        self.episode_rewards.append(reward)

    def get_batch(self):
        index = np.random.randint(0, len(self.states), self.batch_size)

        states = np.array(self.states, dtype=float)[index]
        probs = np.array(self.action_probs, dtype=float)[index]
        rewards = np.array(self.rewards, dtype=float)[index]

        states = T.FloatTensor(states)
        probs = T.FloatTensor(probs)
        rewards = T.FloatTensor(rewards)

        return states, probs, rewards

    def clear_memory(self):
        self.states = []
        self.action_probs = []
        self.rewards = []

    def store_episode(self, reward):
        self.episode_rewards = [reward * R for R in self.episode_rewards]
        #print(self.episode_rewards)

        self.states += self.episode_states
        self.action_probs += self.episode_action_probs
        self.rewards += self.episode_rewards

        self.episode_states = []
        self.episode_action_probs = []
        self.episode_rewards = []

        
class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.actor_network = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic_network = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, 1),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, board):
        state = T.FloatTensor(board).to(self.device)
        return self.actor_network(state), self.critic_network(state)

class Node:
    def __init__(self, prior, to_play, prev_state=None, prev_action=None):
        self.game = ConnectN()

        self.prior = prior
        self.to_play = to_play

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        if prev_state is not None:
            self.state = self.game.get_next_state(prev_state, prev_action, to_play)
        else:
            self.state = self.game.init_state
        
    def expand(self, probs=None):
        if probs is None:
            probs = np.random.uniform(0, 1, self.game.columns)
            probs /= sum(probs)
        for action, prob in enumerate(probs):
            if prob != 0:
                next_state = self.game.get_next_state(self.state, action, -self.to_play)
                self.children[action] = Node(prob, -self.to_play, next_state, action)
    
    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def calc_ucb(self, parent, child):
        actor_weight = child.prior
        if child.visit_count > 0:
            value_weight = -child.value()
        else:
            value_weight = 0
        visit_weight = np.sqrt(parent.visit_count) / (child.visit_count + 1)
        return value_weight + (actor_weight * visit_weight)

    def select_child(self):
        scores = []
        for child in self.children.values():
            scores.append(self.calc_ucb(self, child))
        idx = scores.index(max(scores))
        best_action = list(self.children.keys())[idx]
        best_child = list(self.children.values())[idx]
        return best_action, best_child

    def choose_action(self, temperature):
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
            probs = visit_counts / sum(visit_counts)
        elif temperature == float('inf'):
            probs = np.random.uniform(size=4)
            probs /= sum(probs)
            action = actions[np.argmax(probs)]
        else:
            visit_count_dist = visit_counts ** (1 / temperature)
            visit_count_dist /= sum(visit_count_dist)
            action = np.random.choice(actions, p=visit_count_dist)
            probs = visit_count_dist
        return action, probs


class MCTS:
    def __init__(self, model, n_simulations=5, game=ConnectN()):
        self.n_simulations = n_simulations
        self.game = game
        self.model = model

        '''
        For sim in n_simulations do:
            1) Selection -      Start from root node, use UCB to reach leaf node.
            2) Expansion -      If leaf node is terminal then backprop. Else create
                                n_actions child nodes.
            3) Simulation -     NORMAL SEARCH: From newly expanded child nodes, run policy rollouts
                                (simulations using policy network for both players)
                                and update node value, n_vicoties/n_games_from_node.
                                NEW METHOD: Rather than running simulations to get values of nodes,
                                use value network to estimate value. Rather than 'rollout', this
                                will be reffered to as 'evaluation'.
            4) Backpropogation- Update all nodes selected with result of simulation.
        end
        '''
    def search(self, node=None):
        if node is None:
            node = Node(prior=0, to_play=1)
        if not node.expanded():
            node.expand()
        root = node

        search_path = [node]
        ## Find a leaf node (SELECT)
        while node.expanded():
            _, node = node.select_child()
            search_path.append(node)

        ## EVALUATE
        value = self.game.get_player_reward(node.state, node.to_play)
        if value is None:
            probs, value = self.model.forward(node.state)
            valid_moves = self.game.get_valid_moves(node.state)
            probs *= valid_moves
            probs /= np.sum(probs)

            ## EXPAND
            node.expand(probs)
        
        ## BACKPROPOGATE
        self.backprop(search_path, value)
        return root 

    def backprop(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value * node.to_play
            node.visit_count += 1

    def run(self, node=None):
        for _ in range(self.n_simulations):
            root = self.search(node)
        return root


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


if __name__ == '__main__':
    agent = Agent(lr=1e-4, batch_size=64, fc1_dims=32, fc2_dims=32)
    n_epochs = 500
    learn_frequency = 100
    learning_steps_per_batch = 2

    for epoch in tqdm(range(n_epochs)):
        for game in range(learn_frequency):
            agent.play_game()
        for _ in range(learning_steps_per_batch):
            agent.learn()
        if epoch % 5 == 0:
            agent.memory.clear_memory()



