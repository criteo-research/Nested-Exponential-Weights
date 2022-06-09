import os
from datetime import date

today = date.today()

def get_task_name(params):
    task_name = 'algo:{}'.format(params['algo'])
    if len(params) > 1:
        task_name += '|' + '|'.join('{}:{}'.format(k, v) for k, v in sorted(params.items()) if k not in ['algo', 'experiment'])
    return task_name

def get_metrics_information(metrics):
    metrics_information = ''
    if len(metrics) > 1:
        metrics_information += '|' + '|'.join('{}:{}'.format(k, v) for k, v in sorted(metrics.items()))
    return metrics_information[1:]

def get_results_file_name(params):
    results_dir = os.path.join('results/{}/{}'.format(params['experiment'], today.strftime("%d-%m-%Y")), params['algo'])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return os.path.join(results_dir, 'metrics.txt')


def save_result(settings, regret, reward, round):
    task_name = 'algo:{}'.format(settings['algo'])
    task_name += '|{}:{}'.format('nb_levels', settings['nb_levels'])
    task_name += '|{}:{}'.format('nb_leaves_per_class', settings['nb_leaves_per_class'])
    task_name += '|{}:{}'.format('rd', settings['rd'])
    task_name += '|{}:{}'.format('round', round)

    metrics_information = 'regret:{}'.format(regret)
    metrics_information += '|{}:{}'.format('reward', reward)

    result = '{} {}\n'.format(task_name, metrics_information)

    results_dir = 'results/{}'.format(today.strftime("%d-%m-%Y"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fname = os.path.join(results_dir, 'metrics.txt')

    with open(fname, 'a') as file:
        file.write(result)