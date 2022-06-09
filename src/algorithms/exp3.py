
import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from itertools import combinations

import numpy as np
import time
from tqdm import tqdm
import psutil
from src.utils.save_results import save_result

EPS = 1e-8

class Exp3:

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
        self.action_set = environment.tree.get_all_leaves()
        self.number_of_actions = len(self.action_set)
        # Build a table to store parent nodes for each leaves
        self.paths_action_set = [environment.tree.get_parent_nodes(action_node) for action_node in self.action_set]

    def vector_proba(self, y):
        stable_exp_y = np.exp(y - np.max(y))
        proba_vector = stable_exp_y/np.sum(stable_exp_y)
        return proba_vector

    def sample_action(self, y):
        proba = self.vector_proba(y)
        idx_list = range(self.number_of_actions)
        action = self.rng.choice(idx_list, p=proba)
        return action, proba

    def update_score(self, y, P, a, u):
        # if self.settings['learning_rate'] == 'practical':
        #     learning_rate = 1/np.sqrt(self.max_round)
        # elif self.settings['learning_rate'] == 'theoretical':
        #     learning_rate = np.sqrt(2 * np.log(self.number_of_actions)/(self.number_of_actions*self.max_round))
        # else:
        #     lr = float(self.settings['learning_rate'])
        #     learning_rate = lr / np.sqrt(self.max_round)


        # y[a] = y[a] + learning_rate*u/(P[a]+EPS)
        y[a] = y[a] + u/(P[a]+EPS)

        # y = y - np.max(y)
        return y

    def iterate_learning(self):
        """
        run the agent for "max_rounds" rounds
        """
        initial_dist = np.full(self.number_of_actions, 1.0 / self.number_of_actions)
        score_vector = initial_dist

        metrics = {
            'reward': [],
            'regret': [],
            'round': []
        }
        regrets = []
        rewards = []

        for round in tqdm(range(0, self.max_round)):

            # Choose action
            lr = 1/np.sqrt(round+1)
            action, proba = self.sample_action(score_vector*lr)

            # Receive rewards from environment
            reward = self.environment.get_reward_by_path(self.paths_action_set[action])
            best_strategy_reward = self.environment.get_best_strategy_reward()
            regrets.append(best_strategy_reward - reward)
            rewards.append(reward)

            # Update scores
            score_vector = self.update_score(score_vector, proba, action, reward)

            if round % 100 == 0:
                metrics['reward'].append(np.mean(rewards))
                regret = np.sum(regrets)
                metrics['regret'].append(regret)
                metrics['round'].append(round)
                save_result(self.settings, regret, np.mean(rewards), round)

        # Visualisation
        self.score_vector = score_vector

        return metrics