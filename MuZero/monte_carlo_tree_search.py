import numpy as np

class Node:
    def __init__(self, prior):
        self.prior = prior

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, network, n_simulations, dirichlet_alpha=0.3, exploration_factor=0.25):
        self.n_simulations = n_simulations
        self.discount = 0.997
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_factor = exploration_factor
        self.network = network.load_models()

        '''
        For sim in n_simulations do:
            1) Selection -      Start from root node, use UCB to reach leaf node.
            2) Expansion -      If leaf node is terminal then backprop. Else create
                                n_actions child nodes.
            3) Simulation -     NORMAL SEARCH: From newly expanded child nodes, run policy rollouts
                                (simulations using policy network for both players)
                                and update node value, n_victories/n_games_from_node.
                                NEW METHOD: Rather than running simulations to get values of nodes,
                                use value network to estimate value (unless state is terminal, in 
                                which case value is known). Rather than 'rollout', this
                                will be reffered to as 'evaluation'.
            4) Backpropogation- Update all nodes selected with result of simulation.
        end
        '''

    def expand_node(self, node, probs, hidden_state, reward):
        node.hidden_state = hidden_state
        node.reward = reward
        ## MuZero doesn't mask illegal actions in the tree search
        for action, prob in enumerate(probs):
            node.children[action] = Node(prior=prob)


    def calc_ucb(self, parent, child):
        c1 = 1.25
        c2 = 19652
        x = c1 + np.log((parent.visit_count + c2 + 1) / c2)
        prior_term = parent.prior * x * np.sqrt(parent.visit_count) / (1 + child.visit_count)
        if child.visit_count > 0:
            value_term = -child.value()
        else:
            value_term = 0
        return value_term + prior_term

    def add_exploration_noise(self, node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        factor = self.exploration_factor
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - factor) + n * factor

    def select_child(self, node):
        scores = []
        for child in node.children.values():
            scores.append(self.calc_ucb(node, child))
        idx = scores.index(max(scores))
        best_action = list(node.children.keys())[idx]
        best_child = list(node.children.values())[idx]
        return best_action, best_child

    def choose_action(self, temperature, action_space_size, node):
        visit_counts = np.array([child.visit_count for child in node.children.values()])
        actions = [action for action in node.children.keys()]
        eta = 5e-2

        visit_count_dist = visit_counts ** (1 / (temperature + eta))
        visit_count_dist /= sum(visit_count_dist)
        action = np.random.choice(actions, p=visit_count_dist)

        probs = np.zeros(action_space_size)
        for idx, act in enumerate(actions):
            probs[act] = visit_count_dist[idx]
        return action, probs

    def backprop(self, search_path, value):
        for idx, node in enumerate(reversed(search_path)):
            node.value_sum += value * pow(-1, idx+1)
            node.visit_count += 1
            value = node.reward + self.discount * value

    def search(self, root):
        for _ in range(self.n_simulations):
            node = root
            search_path = [node]
            action_history = []

            ## SELECT (Find a leaf node)
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
                action_history.append(action)
            
            ## EVALUATE
            parent = search_path[-2]
            last_action = action_history[-1]
            probs, value = self.network.actor_critic(parent.hidden_state)
            hidden_state, reward = self.network.roll_forward(parent.hidden_state, last_action)

            ## EXPAND
            self.expand_node(node, probs, hidden_state, reward[0])
            self.add_exploration_noise(node)
            
            ## BACKPROPOGATE
            self.backprop(search_path, value[0])
        return root