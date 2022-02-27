import enum
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

## 1d connect N game
class ConnectN:
    def __init__(self, N=2):
        self.columns = N + 2*(N // 2)
        self.N = N
        self.init_state = valid_moves = np.zeros(self.columns, dtype=int)

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
            elif not self.get_valid_moves(board):
                return None
        return False

    def get_player_reward(self, board, player):
        reward = self.check_terminal(board, player)
        if reward == False:
            return 0
        return reward
    
    def get_board(self, board, player):
        return player * board


class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.actor_network = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim-1)
        )

        self.critic_network = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
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
        state = T.FloatTensor(board.astype(np.float32)).to(self.device)
        return self.actor_network(state), self.critic_network(state)

class Node:
    def __init__(self, prev_state, prev_action, prior, to_play, game=ConnectN()):
        self.game = game

        self.prior = prior
        self.to_play = to_play

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        if prev_state is not None:
            self.state = game.get_next_state(prev_state, prev_action, to_play)
        else:
            self.state = game.init_state
        
    def expand(self, probs):
        for action, prob in enumerate(probs):
            if prob != 0:
                next_state = self.game.get_next_state(self.state, action, -self.to_play)
                self.children[action] = Node(self.state, action, prob, -to_play)
    
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
        best_action, best_child = self.children.items()[idx]
        return best_action, best_child

    def choose_action(self, temperature):
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float('inf'):
            action = np.random.choice(actions)
        else:
            visit_count_dist = visit_counts ** (1 / temperature)
            visit_count_dist = visit_count_dist / sum(visit_count_dist)
            action = np.random.choice(actions, p=visit_count_dist)
        return action


class MCTS:
    def __init__(self, n_simulations=5, game=ConnectN()):
        self.n_simulations = n_simulations
        self.game = game
        self.model = ActorCriticNetwork(lr=0.01, input_dims=[game.columns], \
            fc1_dims=16, fc2_dims=16, n_actions=game.columns)

        '''
        For sim in n_simulations do:
            1) Selection -      Start from root node, use UCB to reach leaf node.
            2) Expansion -      If leaf node is terminal then backprop. Else create
                                n_actions child nodes.
            3) Simulation -     From newly expanded child nodes, run policy rollouts
                                (simulations using policy network for both players)
                                and update node value, n_vicoties/n_games_from_node.
            4) Backpropogation- Update all nodes selected with result of simulation.
        end
        '''
    def select_and_expand(self, node=None):
        if node is None:
            node = Node(prior=0, to_play=1, game=self.game)

        search_path = [node]
        ## Find a leaf node
        while node.expanded():
            _, node = node.select_child()
            search_path.append(node)

        probs, _ = self.model.forward(node.state)
        valid_moves = self.game.get_valid_moves(node.state)
        probs *= valid_moves
        probs /= np.sum(probs)
        node.expand(probs)
        return node.children, search_path

    def rollout(self, node):
        value = None
        while value is None:
            action, node = node.select_child()
            next_state = self.game.get_next_state(node.state, node.to_play, action)
            value = self.game.get_player_reward(next_state, node.to_play)

            probs, value = self.model.forward(next_state)
            valid_moves = self.game.get_valid_moves(next_state)
            probs *= valid_moves
            probs /= np.sum(probs)
            node.expand(probs)
        return value

    def backprop(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value * node.to_play
            node.visit_count += 1

    def search(self, node=None):
        for _ in range(self.n_simulations):
            child_nodes, search_path = self.select_and_expand(node)
            for idx, node in child_nodes.items():
                value = self.rollout(node)
                self.backprop(search_path, value)