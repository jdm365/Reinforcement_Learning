import numpy as np
from games import ConnectN
import sys

class Node:
    def __init__(self, prior, prev_state=None, prev_action=None, game=None):
        self.game = game

        self.prior = prior

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        if prev_state is not None:
            self.state = self.game.get_next_state(prev_state, prev_action)
        else:
            self.state = self.game.init_state

    def expand(self, probs=None):
        if probs is None:
            probs = np.random.uniform(0, 1, self.game.columns)
            probs /= sum(probs)
        for action, prob in enumerate(probs):
            if prob != 0:
                init_state = np.array(self.state, dtype=int)
                self.children[action] = Node(prob, self.state, action, self.game)
                self.state = init_state
    
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

        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
            probs = visit_counts / sum(visit_counts)
        elif temperature == float('inf'):
            probs = np.random.uniform(size=len(actions))
            probs /= sum(probs)
            action = actions[np.argmax(probs)]
        else:
            probs = np.zeros(4, dtype=int)
            visit_count_dist = visit_counts ** (1 / temperature)
            visit_count_dist /= sum(visit_count_dist)
            action = np.random.choice(actions, p=visit_count_dist)
            probs[actions] = visit_count_dist
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
                                use value network to estimate value (unless state is terminal, in 
                                which case value is known). Rather than 'rollout', this
                                will be reffered to as 'evaluation'.
            4) Backpropogation- Update all nodes selected with result of simulation.
        end
        '''

    def search(self, root):
        node = root
        search_path = [node]
        ## Find a leaf node (SELECT)
        while node.expanded():
            _, node = node.select_child()
            search_path.append(node)
        
        ## EVALUATE
        value = self.game.get_reward(node.state)
        if value is None:
            probs, value = self.model.forward(node.state)
            probs = probs.cpu().detach().numpy()
            value = value.cpu().detach().numpy()
            valid_moves = self.game.get_valid_moves(node.state)
            probs *= valid_moves
            probs /= np.sum(probs)

            ## EXPAND
            node.expand(probs)
        
        ## BACKPROPOGATE
        self.backprop(search_path, value)

    def backprop(self, search_path, value):
        for idx, node in enumerate(reversed(search_path)):
            node.value_sum += value * (-1 ** (idx+1))
            node.visit_count += 1

    def run(self, root=None):
        if root is None:
            root = Node(prior=0, game=self.game)
            root.expand()
        for _ in range(self.n_simulations):
            self.search(root)
        return root