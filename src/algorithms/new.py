
import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import numpy as np
from tqdm import tqdm
import time
import psutil
from src.utils.save_results import save_result

EPS = 1e-8

class NestedExponentialWeights:

    def __init__(self, settings):
        """
        :param number_of_actions: number of actions from which the slates will be formed, K.
        :param slate_size: slate size, s.
        :param max_rounds: the number of rounds for which the algorithm will run.
        """

        self.rng = np.random.RandomState(settings['rd'])
        self.max_round = settings['max_rounds']
        self.settings = settings

    def set_environment(self, environment):
        """
        :param environment: this should be a function that can take a vector of size K
        (indicator vector of the chosen slate), and the current round, t as parameters and return the loss/reward
        associated with that slate and that slate only. The indicator vector will have non-zero elements which represent
        the chosen actions in that slate and zero elements which represent actions that are not chosen. The reward/loss
        for actions that are not chosen must be 0, and for the chosen actions the reward/loss should be in [-1,1] or else
        it will be clipped. Hence the output vector must also be a vector of size K with elements clipped to be in [-1,1].
        """
        self.environment = environment

    def vector_proba(self, y):
        """

        Parameters
        ----------
        y

        Returns
        -------

        """
        stable_exp_y = np.exp(y - np.max(y))
        proba_vector = stable_exp_y/np.sum(stable_exp_y)
        return proba_vector

    def sample_node_path(self, round):
        node_path = []
        proba_path = []
        reward_path = []
        node = self.environment.tree.root

        while bool(node.children):
            lr = 1 / np.sqrt(round+1)
            proba = self.vector_proba(node.scores_children*lr)
            idx_list = range(node.nb_children)
            idx_node = self.rng.choice(idx_list, p=proba)
            child_node = node.children[idx_node]
            node_path.append(idx_node)
            reward_child = self.environment.get_reward_by_node(child_node)
            reward_path.append(reward_child)
            proba_path.append(proba[idx_node])
            node = child_node

        return node_path, proba_path, reward_path


    def update_score(self, nodes_path, proba_path, reward_path):

        node = self.environment.tree.root
        level = 0

        proba = 1

        for idx_node, P, reward in zip(nodes_path, proba_path, reward_path):

            # Get joint proba up to level l
            proba *= P
            node.scores_children[idx_node] = node.scores_children[idx_node] + reward / (proba + EPS)

            node = node.children[idx_node]
            level += 1

    def iterate_learning(self):
        """
        run the agent for "max_rounds" rounds
        """
        metrics = {
            'reward': [],
            'regret': [],
            'round': []
        }
        regrets = []
        rewards = []

        for round in tqdm(range(0, self.max_round)):

            # Choose action and receive reward iteratively
            node_path, proba_path, reward_path = self.sample_node_path(round)

            # Reward from environment
            reward = np.sum(reward_path)
            best_strategy_reward = self.environment.get_best_strategy_reward()
            regrets.append(best_strategy_reward - reward)
            rewards.append(reward)

            # Update scores
            self.update_score(node_path, proba_path, reward_path)

            if round % 100 == 0:
                metrics['reward'].append(np.mean(rewards))
                regret = np.sum(regrets)
                metrics['regret'].append(regret)
                metrics['round'].append(round)
                save_result(self.settings, regret, np.mean(rewards), round)

        # Visualization
        self.score_vector = None

        return metrics