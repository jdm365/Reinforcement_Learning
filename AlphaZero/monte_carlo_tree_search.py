import numpy as np
from games import ConnectN

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