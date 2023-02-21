import numpy as np
import torch as T
from games import Connect4


class Node:
    def __init__(self, prior, game, prev_state=None, prev_action=None):
        self.game = game

        self.prior = prior

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.state = self.game.get_next_state(board=prev_state, action=prev_action)


    def expand(self, probs=None):
        '''
        Expand a node by creating child nodes.
        '''
        if probs is None:
            probs = np.random.uniform(0, 1, self.game.columns)
            probs /= sum(probs)

        for action, prob in enumerate(probs):
            #init_state = np.copy(self.state)
            if prob != 0.0:
                self.children[action] = Node(
                        prior=prob, 
                        game=self.game, 
                        prev_state=self.state,
                        prev_action=action
                        )
            #self.state = init_state

    
    def expanded(self):
        '''
        Check if a node has been expanded.
        '''
        return len(self.children) > 0


    def value(self):
        '''
        Calculate the value of a node normalized by the visit count.
        '''
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


    def calc_ucb(self, child):
        '''
        Calculate the Upper Confidence Bound score for a child node.
        UCB = child_value + (child.prior * sqrt(parent.visit_count) / (child.visit_count + 1)))
        '''
        actor_weight = child.prior
        value_weight = -child.value() * (child.visit_count > 0)
        visit_weight = np.sqrt(self.visit_count) / (child.visit_count + 1)
        return value_weight + (actor_weight * visit_weight)


    def select_child(self):
        '''
        Select the child with the highest UCB score.
        '''
        scores = []
        for child in self.children.values():
            scores.append(self.calc_ucb(child))

        idx = scores.index(max(scores))

        best_action = list(self.children.keys())[idx]
        best_child = list(self.children.values())[idx]
        return best_action, best_child


    def choose_action(self, temperature):
        '''
        Choose an action based on the visit counts of the children.
        '''
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]

        probs = list(np.zeros(self.game.columns, dtype=np.float32))
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
        return action, probs


class MCTS:
    def __init__(self, model, n_simulations, game):
        self.n_simulations = n_simulations
        self.game = game
        self.model = model

        '''
        For sim in n_simulations do:
            1) Selection -       Start from root node, use UCB to reach leaf node.
            2) Expansion -       If leaf node is terminal then backprop. Else create
                                 n_actions child nodes.
            3) Simulation -      NORMAL SEARCH: From newly expanded child nodes, run policy rollouts
                                 (simulations using policy network for both players)
                                 and update node value, n_victories/n_games_from_node.
                                 NEW METHOD: Rather than running simulations to get values of nodes,
                                 use value network to estimate value (unless state is terminal, in 
                                 which case value is known). Rather than 'rollout', this
                                 will be referred to as 'evaluation'.
            4) Backpropogation - Update all nodes selected with result of simulation.
        end
        '''

    def search(self, root):
        '''
        Run MCTS search.
        '''
        for _ in range(self.n_simulations):
            ## Always start from the root node
            node = root
            search_path = [node]

            ## Find a leaf node (SELECT)
            while node.expanded():
                _, node = node.select_child()
                search_path.append(node)
            
            ## EVALUATE
            value = self.game.get_reward(node.state)
            if value is None:
                with T.no_grad():
                    probs, value = self.model.forward(node.state)
                probs = probs.cpu().numpy()
                value = value.cpu().numpy()

                ## Mask invalid moves
                valid_moves = self.game.get_valid_moves(node.state)
                probs *= valid_moves
                probs /= np.sum(probs)

                ## EXPAND
                node.expand(probs)
            
            ## BACKPROPOGATE
            self.backprop(search_path, value)
        return root


    def backprop(self, search_path, value):
        '''
        Backpropogate the value of a leaf node to all nodes in the search path.
        '''
        for idx, node in enumerate(reversed(search_path)):
            node.value_sum += value * pow(-1, idx+1)
            node.visit_count += 1


    def run(self, root=None):
        '''
        Run MCTS search and return the root node.
        '''
        if root is None:
            root = Node(prior=0, game=self.game)

        root = self.search(root)
        return root
