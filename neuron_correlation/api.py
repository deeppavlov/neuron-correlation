import copy
import multiprocessing as mp
from typing import Dict

from tensorflow.examples.tutorials.mnist import input_data


def conduct_train_experiment(config: Dict):
    """Launches training of a model accompanied with collecting and
    saving of tensors. Each launch is initiated in separate process.
    The number of launches is config['num_repeats']. Collects metrics
    on test dataset and initiates averaging and saving of results into
    file 'mean_and_std.txt'

    Args:
        config: Dict with the configuration of the experiment
    """
    for i in range(config['num_repeats']):
        repetition_config = prepare_repetition_config(config, i)
        conduct_experiment_repetition_in_sep_proc(repetition_config)
    metrics_mean_and_stddev = calculate_metrics_mean_and_stddev(config['save_path'])
    save_metrics_mean_and_stddev(metrics_mean_and_stddev, config['save_path'])

