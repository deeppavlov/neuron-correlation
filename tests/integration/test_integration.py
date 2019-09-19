import copy
import importlib
import os
import pytest

import numpy as np

import tests.utils_for_testing.dataset as dataset_utils
import tests.utils_for_testing.scheduler as scheduler_utils
from neuron_correlation import api


MNIST_MLP_CONFIG = {
    "num_repeats": 10,
    "save_path": "results_of_tests/integration/test_mnist/train_repeatedly/only_training",
    "graph": {
        "type": "feedforward",
        "input_size": 784,
        "layers": [
            {
                "type": "dense",
                "size": 1000,
                "activation": "relu"
            },
            {
                "type": "dense",
                "size": 10,
                "activation": "softmax"
            }
        ],
        "init_parameter": 1,
        "optimizer": "sgd",
        "metrics": ["accuracy", "perplexity"]
    },
    "train": {
        "tensorflow_seed": None,
        "restore_path": None,
        "validation": {
            "scheduler": {
                "type": "true_every_n_epochs",
                "epochs": 10
            },
            "dataset": {
                "tfds.load": {
                    "name": "mnist:3.*.*",
                    "split": "train[-20%:]",
                    "batch_size": 10000,
                    "as_supervised": True
                }
            }
        },
        "checkpoints": {
            "type": "true_every_n_epochs",
            "epochs": 10
        },
        "stop": {
            "type": "fixed",
            "epochs": 100
        },
        "learning_rate": {
            "type": "exponential_decay",
            "init": 0.3,
            "decay": 0.5,
            "decay_epochs": 10
        },
        "dropout": 0.5,
        "dataset": {
            "tfds.load": {
                "name": "mnist:3.*.*",
                "split": "train[:80%]",
                "batch_size": 32,
                "as_supervised": True
            }
        },
        "print_log": {
            "every_validation"
        }

    },
    "test": {
        "mnist_test": {
            "dataset": {
                "tfds.load": {
                    "name": "mnist:3.*.*",
                    "split": "test",
                    "batch_size": 10000,
                    "as_supervised": True
                }
            }
        }
    }
}


TENSOR_COLLECT_TRAIN_CONFIG = {
    "tensor1": {
        "type": "true_every_n_steps",
        "steps": 1000
    },
    "tensor2": {
        "type": "true_on_logarithmic_scale",
        "init": 0,
        "factor": 1.02
    }
}


SUMMARY_TENSORS_CONFIG = {
    "tensor1": {
        "module": "tensorflow",
        "function": "zeros",
        "args": [[3, 7, 9]]
    },
    "tensor2": {
        "module": "tensorflow",
        "function": "ones",
        "args": [[11, 13, 17]]
    }
}


def get_number_from_file(file_name):
    """Opens file in read mode and tries to convert text inside into float.
    If succeeds return a number else returns `None`

    Args:
        file_name: path like
    Returns:
        float or None
    """
    if os.path.exists(file_name):
        with open(file_name) as f:
            text = f.read()
        try:
            value = float(text)
        except ValueError:
            value = None
    else:
        value = None
    return value


def get_summary_tensors_values(config):
    """Evaluates simple tensors from summary tensors creation configs"""
    values = {}
    for k, v in config.items():
        module = importlib.import_module(v['module'])
        func = getattr(module, v['function'])
        values[k] = func(*v['args'])
    return values


def check_summarized_tensors_in_dir(dir_, tensor_values, tensor_steps):
    report = {}
    dirs = [e for e in os.listdir(dir_) if os.path.isdir(e)]
    for d in dirs:
        if d not in tensor_values:
            report[d] = None
            report['ok'] = False
    # TODO


def get_number_of_steps(stop_config, dataset_config):
    batch_size = dataset_config['tfds.load']['batch_size']
    dataset_size = dataset_utils.get_dataset_size(dataset_config)
    if stop_config['type'] == 'fixed':
        if 'steps' in stop_config:
            number_of_steps = stop_config['steps']
        elif 'epochs' in stop_config:
            number_of_steps = np.ceil(dataset_size / batch_size) * stop_config['epochs']
        else:
            raise ValueError(
                "Only stop configs with 'epochs' or 'steps'"
                " items are supported.\nStop config:\n{}".format(stop_config)
            )
    else:
        raise ValueError("Only stop configs of type 'fixed' are supported")
    return number_of_steps


def get_true_scheduler_steps(config, num_steps):
    if config['type'] == 'true_every_n_steps':
        steps = list(range(0, num_steps, config['steps']))
    elif config['type'] == 'true_on_logarithmic_scale':
        steps = scheduler_utils.logarithmic_int_range(config['init'], num_steps, config['factor'])
    else:
        raise ValueError(
            "Unsupported Scheduler config type:\n'{}'".format(config['type']))
    return steps


def get_summary_tensors_steps(tensors_config, stop_config, dataset_config):
    num_steps = get_number_of_steps(stop_config, dataset_config)
    steps = {}
    for tensor, t_config in tensors_config.items():
        steps[tensor] = get_true_scheduler_steps(t_config, num_steps)
    return steps


class TestTrainRepeatedly:
    def test_training_without_tensor_saving(self):
        """Check saved loss value on test dataset"""
        save_path = "results_of_tests/integration/test_mnist/train_repeatedly/only_training"
        config = copy.deepcopy(MNIST_MLP_CONFIG)
        config['save_path'] = save_path

        api.train_repeatedly(config)

        test_loss_file_name = save_path + '/testing/mnist_test/loss.txt'
        loss = get_number_from_file(test_loss_file_name)
        assert loss is not None, \
            "loss file name {} is not found or broken".format(test_loss_file_name)
        assert 0.95 <= loss <= 1.0, \
            "average loss {} on test dataset does not belong to interval {}".format(loss, [0.95, 1.0])

    def test_train_tensor_saving(self):
        """Check quantity, shape and values
        of summarized tensors and steps on which
        they were collected.
        """
        save_path = "results_of_tests/integration/test_mnist/train_repeatedly/train_tensor_saving"
        config = copy.deepcopy(MNIST_MLP_CONFIG)
        config['save_path'] = save_path

        config['train']['train_summary_tensors'] = copy.deepcopy(TENSOR_COLLECT_TRAIN_CONFIG)
        config['graph']['summary_tensors'] = copy.deepcopy(SUMMARY_TENSORS_CONFIG)

        api.train_repeatedly(config)

        dirs_with_tensors = [save_path + '/{}/train_tensors'.format(i) for i in range(config['num_repeats'])]
        train_tensors_creation_config = {
            k: v for k, v in config['graph']['summary_tensors'].items()
            if k in config['train']['train_summary_tensors']
        }
        tensor_values = get_summary_tensors_values(train_tensors_creation_config)
        tensor_steps = get_summary_tensors_steps(config['train'])
        for dir_ in dirs_with_tensors:
            report = check_summarized_tensors_in_dir(dir_, tensor_values, tensor_steps)
            assert report['ok'], "Summarized tensors in directory {} are not ok. " \
                                 "Report:\n{}\nconfig['graph']['summary_tensors']:\n{}" \
                                 "\nconfig['train']['train_summary_tensors']:\n{}".format(
                dir_,
                report,
                config['graph']['summary_tensors'],
                config['train']['train_summary_tensors']
            )


