import numpy as np
from collections import deque
from src.environment.tree import Tree, Node


class Environment:

    def __init__(self, settings):

        """
        :param number_of_actions: K.
        :param round_no: max_rounds, time horizon.
        :param slate_size: s.
        """
        self.sampling_rng = np.random.RandomState(settings['rd'])
        self.env_rng = np.random.RandomState(123)

        self.nb_leaves_per_class = settings['nb_leaves_per_class']
        self.nb_levels = settings['nb_levels']

        self._initialize_tree()

    def _initialize_tree(self):
        self.root = None
        self.tree = Tree()
        self.root = self.tree.insert(self.root, ('root', 0))

    def set(self):
        self.iterative_graph_create()
        _, self.best_strategy_path = self.tree.find_best_arm_path()
        nodes_path = [self.root]
        node = self.root
        for idx in self.best_strategy_path:
            nodes_path.append(node.children[idx])
            node = node.children[idx]
        self.best_strategy_nodes_path = nodes_path

    def get_reward_by_node(self, node):
        level_correction = 10 ** (-node.level)
        h = 0.1 * level_correction
        low = node.value - h/2
        high = node.value + h/2
        return self.sampling_rng.uniform(low=low, high=high)

    def get_R_level(self, level):
        return 10 * 10 ** (-level)

    def get_reward_by_path(self, path):
        path_wo_root = path[1:]
        return np.sum([self.get_reward_by_node(node) for node in path_wo_root])

    def get_best_strategy_reward(self):
        return self.get_reward_by_path(self.best_strategy_nodes_path)

    def create_sub_level(self, node_key):
        node = self.tree.graph[node_key]
        level = node.level + 1
        for n in range(self.nb_leaves_per_class):
            name = 'level_{}_child_number_{}_of_{}'.format(level, n, node.name)
            value = self.env_rng.choice(range(1, 9))
            data = name, value
            _, node = self.tree.insert(node, data)

    def iterative_graph_create(self, start=None):
        """
        Creates graph by breadth
        Parameters
        ----------
        start

        Returns
        -------

        """
        if not start:
            start = 'root'

        visited = []
        queue = deque()
        queue.append(start)

        while queue:
            node_key = queue.popleft()
            if node_key not in visited:
                visited.append(node_key)
                self.create_sub_level(node_key)
                unvisited = [n.name for n in self.tree.graph[node_key].children if
                             (n.name not in visited and n.level < self.nb_levels)]
                queue.extend(unvisited)

        return visited


class BlueBusRedBusEnvironment(Environment):

    def __init__(self, settings):
        """
        :param number_of_actions: K.
        :param round_no: max_rounds, time horizon.
        :param slate_size: s.
        """
        super(BlueBusRedBusEnvironment, self).__init__(settings)

        self._initialize_tree()
        self.nb_assymetric_child = settings['nb_leaves_per_class']

    def set(self):
        self.create_graph()
        _, self.best_strategy_path = self.tree.find_best_arm_path()

        nodes_path = [self.root]
        node = self.root

        for idx in self.best_strategy_path:
            nodes_path.append(node.children[idx])
            node = node.children[idx]

        self.best_strategy_nodes_path = nodes_path

    def _create_sub_level(self, node_key, nb_childs):
        node = self.tree.graph[node_key]
        level = node.level + 1
        for n in range(nb_childs):
            name = 'level_{}_child_number_{}_of_{}'.format(level, n, node.name)
            value = self.env_rng.choice(range(1, 9))
            data = name, value
            _, node = self.tree.insert(node, data)

    def create_graph(self):
        self._create_sub_level('root', 2)
        node_key = 'level_1_child_number_1_of_root'
        self._create_sub_level(node_key, self.nb_assymetric_child)
