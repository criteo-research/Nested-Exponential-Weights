import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

choices_algo = {'exp3': 'Exp3',
                'new': 'NestedExponentialWeights'}

def get_algo_by_name(settings):
    """ Gets distribution according to hyperparameter choice

    Args:
        hyperparams (dic): dictionary of hyperparameters

    """
    name = settings['algo']
    algo_name = "src.algorithms.{}".format(name)
    mod = __import__(algo_name, fromlist=[choices_algo[name]])
    return getattr(mod, choices_algo[name])(settings)

choices_env = {'general': 'Environment',
                'paradox': 'BlueBusRedBusEnvironment'}

def get_env_by_name(settings):
    """ Gets distribution according to hyperparameter choice

    Args:
        hyperparams (dic): dictionary of hyperparameters

    """
    name = settings['env']
    env_path = "src.environment.env"
    mod = __import__(env_path, fromlist=[choices_env[name]])
    return getattr(mod, choices_env[name])(settings)