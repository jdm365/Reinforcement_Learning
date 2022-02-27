import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

## 1d connect N game
class ConnectN:
    def __init__(self, N=2):
        self.columns = N + 2*(N // 2)
        self.N = N
        self.initial_board = np.zeros(N, dtype=int)

    def get_next_state(self, board, player, action):
        board_ = np.copy(board)
        board_[action] = player
        return board_, -player

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
            if victory:
                return True
        return False

    def get_player_reward(self, board, player):
        if self.check_terminal(board, player):
            return 1
        if self.check_terminal(board, -player):
            return -1
        if self.get_valid_moves(board) != False:
            return None
        return 0

    def get_canonical_board(self, board, player):
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
    def __init__(self, prior, to_play):
        self.prior = prior
        self.to_play = to_play

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.state = None

    def __repr__(self):
        prior = '{0:.2f}'.format(self.prior)
        return '{} Prior: {} Count: {} Value: {}'.format(self.state.__str__(), \
                prior, self.visit_count, self.value())

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, probs, state, to_play):
        self.state = state
        for action, prob in enumerate(probs):
            if prob != 0:
                self.children[action] = Node(prior=prob, to_play=-to_play)
    
    def expanded(self):
        return len(self.children) > 0

    def calc_ucb(self, parent, child):
        actor_weight = child.prior
        if child.visit_count > 0:
            value_weight = -child.value()
        else:
            value_weight = 0
        visit_weight = np.sqrt(parent.visit_count) / (child.visit_count + 1)
        return value_weight + (actor_weight * visit_weight)

    def select_child(self):
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = self.calc_ucb(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
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

    def search(self, state, to_play):
        root = Node(prior=0, to_play=to_play)

        probs, value = self.model.forward(state)
        valid_moves = self.game.get_valid_moves(state)
        probs *= valid_moves
        probs /= np.sum(probs)
        self.root.expand(probs, state, to_play)

        for _ in range(self.n_simulations):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state

            next_state, _ = self.game.get_next_state(state, player=1, action=action)
            next_state = self.game.get_canonical_board(next_state, player=-1)

            value = self.game.get_player_reward(player=1)
            if value is None:
                probs, value = self.model.forward(next_state)
                valid_moves = self.game.get_valid_moves(next_state)
                probs *= valid_moves
                probs /= np.sum(probs)
                node.expand(probs, next_state, -parent.to_play)

            self.learn(search_path, value, -parent.to_play)
        return root

    def learn(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1