import copy
import importlib
import multiprocessing as mp
import os
import pickle
import re

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

import tests.utils_for_testing.dataset as dataset_utils
import tests.utils_for_testing.scheduler as scheduler_utils
import tests.utils_for_testing.tf_utils as tf_utils
from neuron_correlation import api
from tests.utils_for_testing.path import get_repo_root


EVENT_FILE_PATTERN = re.compile(r'^events\.out\.tfevents.[0-9]{10}\.')


MNIST_MLP_CONFIG = {
    "num_repeats": 10,
    "save_path": os.path.join(
        get_repo_root(),
        "results_of_tests",
        "integration",
        "test_mnist",
        "train_repeatedly",
        "only_training"
    ),
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
                "counter": 'epoch',
                "type": "true_on_range",
                "step": 10
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
            "counter": "epoch",
            "type": "true_on_range",
            "step": 10
        },
        "stop": {
            "counter": "epoch",
            "type": "true_on_value",
            "value": 100
        },
        "learning_rate": {
            "counter": "epoch",
            "type": "exponential_decay",
            "init": 0.3,
            "decay": 0.5,
            "step": 10
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


TRAIN_TENSOR_SUMMARY_CONFIG = {
    "tensor1": {
        "counter": "step",
        "type": "true_on_range",
        "step": 1000
    },
    "tensor2": {
        "counter": "step",
        "type": "true_on_logrange",
        "init": 0,
        "factor": 1.02,
        "include_stop": True
    }
}


DATASET_TENSOR_SUMMARY_CONFIG = {
    "mnist_test": {
        "dataset": {
            "tfds.load": {
                "name": "mnist:3.*.*",
                "split": "test",
                "batch_size": 10000,
                "as_supervised": True
            }
        },
        "mean": {
            "tensor1": {
                "train_steps": TRAIN_TENSOR_SUMMARY_CONFIG["tensor1"],
                "numbers_of_batches": {
                    "counter": "step",
                    "type": "true_on_range",
                    "step": 1
                }
            }
        },
        "all": {
            "tensor2": {
                "train_steps": TRAIN_TENSOR_SUMMARY_CONFIG["tensor2"],
                "numbers_of_batches": {
                    "counter": "step",
                    "type": "true_on_range",
                    "step": 3
                }
            }
        }
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
    If succeeds, returns a number else returns `None`

    Args:
        file_name: path like
    Returns:
        number: float or None
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
    """Evaluates simple tensors from summary tensors creation configs.
    Works only for tensors which are created independently from neural
    network graph.

    Args:
        config: a dictionary. It is config for tensor creation, passed in 'graph'
            item of experiment config. The keys are tensor names (not in sense of
            tensorflow tensor names) and values are dictionaries with items
            'module', 'function', 'args'.
    Returns:
        values: a dictionary. Keys of the dictionary are tensor names
            and values of the dictionary aretensor values.
    """
    values = {}
    for k, v in config.items():
        module = importlib.import_module(v['module'])
        func = getattr(module, v['function'])
        q = mp.Queue()
        p = mp.Process(target=tf_utils.evaluate_tensor_in_sep_process, args=(q, func, v['args']))
        p.start()
        values[k] = q.get()
        p.join()
    return values


def get_list_of_dataset_tensors_names(config):
    names = []
    for ds_config in config.values():
        for type_ in ['all', 'mean']:
            for name in ds_config[type_].keys():
                if name not in names:
                    names.append(name)
    return names


def check_summarized_tensor(dir_, tensor_name, value, steps):
    report = {'ok': True}
    event_files = [e for e in os.listdir(dir_) if EVENT_FILE_PATTERN.match(e) is not None]
    # steps on which tensor was summarized
    summarized_steps = []
    # all tuples (step, tensor_value) in which tensor_value != value
    wrong_results = []
    for ef in event_files:
        for e in tf.train.summary_iterator(os.path.join(dir_, ef)):
            # `step` is added to `summarized_steps` after confirmation
            # of presence of the required tensor in an event file
            step = e.step
            for v in e.summary.value:
                if v.tag == tensor_name:
                    summarized_steps.append(step)
                    summarized = tf.make_ndarray(v.tensor)
                    if np.any(value != summarized):
                        wrong_results.append((step, summarized))
                        report['ok'] = False
    missing_steps = [s for s in steps if s not in summarized_steps]
    unwanted_steps = [s for s in summarized_steps if s not in steps]
    if missing_steps:
        report['ok'] = False
        report['missing_steps'] = missing_steps
    else:
        report['missing_steps'] = []
    if unwanted_steps:
        report['ok'] = False
        report['unwanted_steps'] = unwanted_steps
    else:
        report['unwanted_steps'] = []
    if wrong_results:
        report['wrong_results'] = wrong_results
    else:
        report['wrong_results'] = []
    report['expected_value'] = value
    report['expected_shape'] = value.shape
    return report


def check_summarized_tensors_in_dir(dir_, tensor_values, tensor_steps):
    report = {'ok': True, 'tensors': {}}
    dirs = [e for e in os.listdir(dir_) if os.path.isdir(os.path.join(dir_, e))]
    for d in dirs:
        if d not in tensor_values:
            report['tensors'][d] = None
            report['ok'] = False
        else:
            dir_report = check_summarized_tensor(
                os.path.join(dir_, d),
                d,
                tensor_values[d],
                tensor_steps[d]
            )
            report['ok'] = report['ok'] and dir_report['ok']
            report['tensors'][d] = dir_report
    return report


def get_number_of_steps(stop_config, dataset_config):
    batch_size = dataset_config['tfds.load']['batch_size']
    dataset_size = dataset_utils.get_dataset_size(dataset_config)
    if stop_config['type'] == 'true_on_value':
        if stop_config['counter'] == 'step':
            number_of_steps = stop_config['value']
        elif stop_config['counter'] == 'epoch':
            number_of_steps = np.ceil(dataset_size / batch_size) * stop_config['value']
        else:
            raise ValueError(
                "Only stop configs with 'epochs' or 'steps'"
                " items are supported.\nStop config:\n{}".format(stop_config)
            )
    else:
        raise ValueError("Only stop configs of type 'fixed' are supported")
    return int(number_of_steps)


def get_true_scheduler_steps(config, num_steps):
    if config['type'] == 'true_on_range':
        steps = list(range(0, num_steps+1, config['step']))
    elif config['type'] == 'true_on_logrange':
        steps = scheduler_utils.logarithmic_int_range(
            config['init'],
            num_steps,
            config['factor'],
            include_stop=config["include_stop"],
        )
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


def make_short_report(report):
    report = copy.deepcopy(report)
    for dir_report in report.values():
        if isinstance(dir_report, dict):
            for v in report['tensors'].values():
                v['number_of_missing_steps'] = len(v['missing_steps'])
                del v['missing_steps']
                v['number_of_unwanted_steps'] = len(v['unwanted_steps'])
                del v['unwanted_steps']
                v['number_of_wrong_results'] = len(v['wrong_results'])
                if v['wrong_results']:
                    v['zeroth_wrong_result'] = v['wrong_results'][0]
                    v['shape_of_zeroth_wrong_result'] = v['wrong_results'][0][1].shape
                del v['wrong_results']
    return report


class TestTrainRepeatedly:
    def test_training_without_tensor_saving(self):
        """Check saved loss value on test dataset"""
        save_path = os.path.join(
            get_repo_root(),
            "results_of_tests",
            "integration",
            "test_mnist",
            "train_repeatedly",
            "only_training"
        )
        config = copy.deepcopy(MNIST_MLP_CONFIG)
        config['save_path'] = save_path

        api.train_repeatedly(config)

        test_loss_file_name = os.path.join(
            save_path,
            'testing',
            'mnist_test',
            'loss.txt'
        )
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
        save_path = os.path.join(
            get_repo_root(),
            "results_of_tests",
            "integration",
            "test_mnist",
            "train_repeatedly",
            "train_tensor_summary"
        )
        config = copy.deepcopy(MNIST_MLP_CONFIG)
        config['save_path'] = save_path

        config['train']['train_summary_tensors'] = copy.deepcopy(TRAIN_TENSOR_SUMMARY_CONFIG)
        config['graph']['summary_tensors'] = copy.deepcopy(SUMMARY_TENSORS_CONFIG)

        api.train_repeatedly(config)
        launches_dirs = [os.path.join(save_path, '{}').format(i) for i in range(config['num_repeats'])]
        train_tensors_creation_config = {
            k: v for k, v in config['graph']['summary_tensors'].items()
            if k in config['train']['train_summary_tensors']
        }
        tensor_values = get_summary_tensors_values(train_tensors_creation_config)
        tensor_steps = get_summary_tensors_steps(
            config['train']['train_summary_tensors'],
            config['train']['stop'],
            config['train']['dataset'],
        )
        reports = []
        for dir_ in launches_dirs:
            report = check_summarized_tensors_in_dir(
                os.path.join(dir_, 'train_tensors'), tensor_values, tensor_steps)
            reports.append(report)
            with open(os.path.join(dir_, 'tensors_report.pickle'), 'wb') as f:
                pickle.dump(report, f)
        for i, (dir_, report) in enumerate(zip(launches_dirs, reports)):
            assert report['ok'], "Summarized tensors in directory '{}' are not ok. " \
                                 "Short report:\n{}\n**********\nFull report is in\n'{}'\n**********\n" \
                                 "config['graph']['summary_tensors']:\n{}\n**********" \
                                 "\nconfig['train']['train_summary_tensors']:\n{}".format(
                os.path.join(dir_, 'train_tensors'),
                make_short_report(report),
                os.path.join(dir_, 'tensors_report.pickle'),
                config['graph']['summary_tensors'],
                config['train']['train_summary_tensors']
            )

    def test_dataset_tensor_saving(self):
        """Check quantity, shape and values
        of summarized tensors and steps on
        which they their collected
        """
        save_path = os.path.join(
            get_repo_root(),
            "results_of_tests",
            "integration",
            "test_mnist",
            "train_repeatedly",
            "dataset_tensor_summary"
        )
        config = copy.deepcopy(MNIST_MLP_CONFIG)
        config['save_path'] = save_path

        config['train']['dataset_summary_tensors'] = copy.deepcopy(DATASET_TENSOR_SUMMARY_CONFIG)
        config['graph']['summary_tensors'] = copy.deepcopy(SUMMARY_TENSORS_CONFIG)

        api.train_repeatedly(config)
        launches_dirs = [os.path.join(save_path, '{}').format(i) for i in range(config['num_repeats'])]
        summarized_tensors_names = get_list_of_dataset_tensors_names(
            config['train']['dataset_summary_tensors'])
        dataset_tensors_creation_config = {
            k: v for k, v in config['graph']['summary_tensors'].items()
            if k in summarized_tensors_names
        }
        tensor_values = get_summary_tensors_values(dataset_tensors_creation_config)
        # TODO
