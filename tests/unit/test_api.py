from unittest import mock

import pytest

from neuron_correlation import api


class TestConductTrainExperiment:
    def test_process_start_join_calls(self):
        config = {
            "num_repeats": 7,
            "graph": {},
            "train": {},
            "test": {}
        }
        with mock.patch('multiprocessing.Process') as MockProcess, \
                mock.patch('multiprocessing.Queue'):
            api.conduct_train_experiment(config)
            start_call_count = MockProcess.return_value.start.call_count
            join_call_count = MockProcess.return_value.join.call_count
            assert start_call_count == config['num_repeats'], \
                "mp.Process.start() was called {} times whereas {} calls were expected".format(
                    start_call_count, config['num_repeats'])
            assert join_call_count == config['num_repeats'], \
                "mp.Process.join() was called {} times whereas {} calls were expected".format(
                    join_call_count, config['num_repeats'])
