# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import argparse

from src.utils.utils import get_algo_by_name, get_env_by_name
from datetime import date
from tqdm import tqdm
today = date.today()

def do_single_experiment(rd, settings):
    env = get_env_by_name(settings)
    env.set()
    agent = get_algo_by_name(settings)
    agent.set_environment(env)
    metrics = agent.iterate_learning()

def main(args):

    NUMBER_RD = 5

    for rd in range(NUMBER_RD):

        settings = {
            'rd': rd,
            'max_rounds': args.max_rounds,
            'env': args.env,
            'nb_leaves_per_class': args.nb_leaves_per_class,
            'nb_levels': args.nb_levels,
            'algo': args.algo,
            'reward_decay': args.r_decay
        }
        do_single_experiment(rd, settings)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scripts for the evaluation of methods')
    parser.add_argument('--max_rounds', nargs="?", type=int, default=1000, help='maximum number of rounds')
    parser.add_argument('--env', nargs="?", default='paradox',  choices=['paradox', 'general'], help='environment')
    parser.add_argument('--nb_levels', nargs="?", type=int, default=3, help='number of levels in the problem')
    parser.add_argument('--nb_leaves_per_class', nargs="?", type=int, default=3, help='number of leaves per class')
    parser.add_argument('--algo', nargs="?", default='exp3',  choices=['exp3', 'nexp3'],
                        help='algo method')
    parser.add_argument('--r_decay', nargs="?", default=10, help='reward decay')
    main(parser.parse_args())
