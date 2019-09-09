import copy
import json
import multiprocessing as mp
import os
from typing import Dict, Union

from tensorflow.examples.tutorials.mnist import input_data


def calculate_metrics_stats(save_path):
    pass


def train_once(config):
    pass


def train_repeatedly(config: Dict):
    """Trains a model several times. For every train session the model is created
    in a separate process. The number of repeats is config['num_repeats'].
    Calculates metrics on test dataset and saves them into file 'metrics_stats.json'

    Args:
        config: a dictionary
    """
    for i in range(config['num_repeats']):
        config_for_repeat = copy.deepcopy(config)
        del config_for_repeat['num_repeats']
        config_for_repeat['save_path'] = os.path.join(config_for_repeat['save_path'], str(i))
        p = mp.Process(train_once, args=(config,))
        p.start()
        p.join()
    metrics_stats = calculate_metrics_stats(config['save_path'])
    metrics_save_path = os.path.join(config['save_path'], 'metrics_stats.json')
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_stats, f)

