import numpy as np
from collections import deque

rng = np.random.RandomState(42)

class Node:
    """
    Class Node
    """

    def __init__(self, data, parent=None):
        """

        Parameters
        ----------
        data
        parent
        """

        self.name, self.value = data
        self.children = []
        self.scores_children = np.array([])
        self.nb_children = 0
        self.parent = parent
        if self.parent:
            self.level = self.parent.level + 1
        else:
            self.level = 0
        level_correction = 10 ** (-self.level)
        self.value = self.value * level_correction

    def get_child_nodes(self, nodes=None):
        """

        Parameters
        ----------
        nodes

        Returns
        -------

        """
        if not nodes:
            nodes = []
        for child in self.children:
            if child.children:
                child.get_child_nodes(nodes)
                nodes.append((child.name, child.value, child.level, child.nb_children))
            else:
                nodes.append((child.name, child.value, child.level, child.nb_children))
        return nodes


class Tree:
    """
    Class tree will provide a tree as well as utility functions.
    """

    def __init__(self):
        """
        """
        self.levels = [[]]
        self.graph = {
            'root': None,
        }
        self.max_level = 0

    def create_node(self, data, parent=None):
        """
        Utility function to create a node.
        """
        return Node(data, parent)

    def insert(self, parent_node, data):
        """
        Insert function will insert a node into tree.
        Duplicate keys are not allowed.
        """
        # if tree is empty , return a root node
        name, value = data
        if parent_node is None:
            node = self.create_node(data)
            if node.level == 0:
                self.root = node
                self.graph['root'] = node
            return node

        node = self.create_node(data, parent_node)
        self.graph[name] = node
        parent_node.children.append(node)
        parent_node.nb_children = len(parent_node.children)
        parent_node.scores_children = np.full(parent_node.nb_children, 1.0 / parent_node.nb_children)

        return node, parent_node

    def get_parent_nodes(self, node):
        """ Get parent nodes of a current node

        Parameters
        ----------
        node
        nodes list on which the recursion is made

        Returns
        -------

        """
        nodes = [node]

        def _recursive_parent_nodes(node, nodes):
            if node.parent:
                nodes.append(node.parent)
                _recursive_parent_nodes(node.parent, nodes)
            return nodes

        nodes = _recursive_parent_nodes(node, nodes)
        nodes.reverse()
        return nodes

    def get_all_nodes(self):
        """ Get all node utility function

        Returns all nodes of the graph
        -------

        """
        nodes = [self.root]

        def _get_nodes(node):
            for child in node.children:
                nodes.append(child)
                _get_nodes(child)

        _get_nodes(self.root)

        nodes = sorted(nodes, key=lambda node: node.level)
        nodes_names = [node.name for node in nodes]

        return nodes

    def get_all_leaves(self):
        """

        Returns
        -------

        """

        leaves = []

        def get_leaves(node):
            # print(node.name, node.children)
            if len(node.children) == 0:
                leaves.append(node)
            if node is not None:
                for child in node.children:
                    get_leaves(child)

        get_leaves(self.root)

        return leaves


    def iterative_dfs(self, node_key=None):
        """" Depth-first search

        Parameters
        ----------
        node

        Returns
        -------

        """
        if not node_key:
            node_key = 'root'
        visited = []
        stack = deque()
        stack.append(node_key)

        while stack:
            node_key = stack.pop()
            if node_key not in visited:
                visited.append(node_key)
                unvisited = [n.name for n in self.graph[node_key].children if n.name not in visited]
                stack.extend(unvisited)

        return visited

    def iterative_bfs(self, start=None):
        """
        Breadth-First Search
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
                unvisited = [n.name for n in self.graph[node_key].children if n.name not in visited]
                queue.extend(unvisited)

        return visited

    def find_max_sum_path(self, root, max_result=-np.infty, max_path=[]):

        # Base Case
        if root is None:
            return 0, max_result

        max_sums = [0]
        max_paths = [[]]
        for child in root.children:
            max_sum, idx_max_path = self.find_max_sum_path(child, max_result, max_path)
            max_sums.append(max_sum)
            max_paths.append(idx_max_path)

        sums = root.value + np.array(max_sums)
        idx = np.argmax(sums)
        max_result = sums[idx]
        max_path = max_paths[idx]
        max_path.append(idx)

        return max_result, max_path

    def find_best_arm_path(self):
        max_mean, path = self.find_max_sum_path(self.root)
        path = np.array(path[1:])
        path = list(path - 1)[::-1]
        return max_mean, path