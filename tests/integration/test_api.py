import os

import pytest

from neuron_correlation import api


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


class TestConductTrainExperiment:
    def test_training_without_tensor_saving(self):
        """Check save loss value on test dataset"""
        save_path = "results_of_tests/neuron_correlation/test_mnist/TestMain/test_only_training"
        config = {
            "num_repeats": 10,
            "tensorflow_seed": None,
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
                "restore_path": None,
                "save_path": save_path,
                "validation": {
                    "scheduler": {
                        "type": "true_every_n_epochs",
                        "epochs": 10
                    },
                    "dataset": {
                        "type": "mnist",
                        "input_file_name": "train-images-idx3-ubyte.gz",
                        "label_file_name": "train-labels-idx1-ubyte.gz",
                        "batch_size": 10000,
                        "first_example_index": 40000,
                        "total_number_of_examples": 10000
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
                    "type": "mnist",
                    "input_file_name": "train-images-idx3-ubyte.gz",
                    "label_file_name": "train-labels-idx1-ubyte.gz",
                    "batch_size": 32,
                    "first_example_index": 0,
                    "total_number_of_examples": 40000
                },
                "print_log": {
                    "every_validation"
                }

            },
            "test": {
                "mnist_test": {
                    "dataset": {
                        "type": "mnist",
                        "input_file_name": "test-images-idx3-ubyte.gz",
                        "label_file_name": "test-labels-idx1-ubyte.gz",
                        "batch_size": 32,
                        "first_example_index": 0,
                        "total_number_of_examples": 10000
                    }
                }
            }
        }

        api.conduct_train_experiment(config)

        test_loss_file_name = save_path + '/testing/mnist_test/loss.txt'
        loss = get_number_from_file(test_loss_file_name)
        assert loss is not None, \
            "loss file name {} is not found or broken".format(test_loss_file_name)
        assert 0.95 <= loss <= 1.0, \
            "average loss {} on test dataset does not belong to interval {}".format(loss, [0.95, 1.0])
