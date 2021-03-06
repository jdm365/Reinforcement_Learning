import numpy as np
import torch as T

class Node:
    def __init__(self, prior, prev_board=None, prev_action=None, game=None):
        self.game = game

        self.prior = prior

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.state, self.board = self.game.get_next_state(prev_board, prev_action)

    def reset_board(self, board):
        self.board = board

    def expand(self, probs=None):
        if probs is None:
            probs = np.random.uniform(0, 1, self.game.n_actions)
            probs /= sum(probs)
        for action, prob in enumerate(probs):
            init_board = self.board.copy()
            if prob != 0:
                action_array = np.zeros(8*8*82, dtype=int)
                action_array[action] = 1
                action_array = action_array.reshape(8, 8, 82)
                self.children[action] = Node(prob, self.board, action_array, self.game)
            self.reset_board(init_board)
    
    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def calc_ucb(self, child):
        actor_weight = child.prior
        if child.visit_count > 0:
            value_weight = -child.value()
        else:
            value_weight = 0
        visit_weight = np.sqrt(self.visit_count) / (child.visit_count + 1)
        return value_weight + (actor_weight * visit_weight)

    def select_child(self):
        scores = []
        for child in self.children.values():
            scores.append(self.calc_ucb(child))
        idx = scores.index(max(scores))
        best_action = list(self.children.keys())[idx]
        best_child = list(self.children.values())[idx]
        return best_action, best_child

    def choose_action(self, temperature):
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]

        probs = list(np.zeros(self.game.n_actions, dtype=int))
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
            visit_count_dist = visit_counts / sum(visit_counts)
            for idx, act in enumerate(actions):
                probs[act] = visit_count_dist[idx]
            probs = np.array(probs)
        else:
            visit_count_dist = visit_counts ** (1 / temperature)
            visit_count_dist /= sum(visit_count_dist)
            action = np.random.choice(actions, p=visit_count_dist)
            for idx, act in enumerate(actions):
                probs[act] = visit_count_dist[idx]
            probs = np.array(probs)

            action_array = np.zeros(8*8*82, dtype=int)
            action_array[action] = 1
            action_array = action_array.reshape(8, 8, 82)
        return action_array, probs


class MCTS:
    def __init__(self, model, n_simulations, game):
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
                                use value network to estimate value (unless state is terminal, in 
                                which case value is known). Rather than 'rollout', this
                                will be reffered to as 'evaluation'.
            4) Backpropogation- Update all nodes selected with result of simulation.
        end
        '''

    def search(self, root):
        for _ in range(self.n_simulations):
            node = root
            search_path = [node]
            ## Find a leaf node (SELECT)
            while node.expanded():
                _, node = node.select_child()
                search_path.append(node)
            
            ## EVALUATE
            value = self.game.get_reward(node.board)
            if value is None:
                self.model.eval()
                with T.cuda.amp.autocast():
                    probs, value = self.model.forward(node.state)
                self.model.train()
                probs = probs.cpu().detach().numpy()
                value = value.cpu().detach().numpy()
                valid_moves = self.game.get_valid_moves(node.board)
                probs *= valid_moves.flatten()
                probs /= np.sum(probs)

                ## EXPAND
                node.expand(probs)
            
            ## BACKPROPOGATE
            self.backprop(search_path, value)
        return root

    def backprop(self, search_path, value):
        for idx, node in enumerate(reversed(search_path)):
            node.value_sum += value * pow(-1, idx+1)
            node.visit_count += 1

    def run(self, root=None):
        if root is None:
            root = Node(prior=0, game=self.game)
        root = self.search(root)
        return root