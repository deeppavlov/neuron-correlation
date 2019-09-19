import copy
import os

import tensorflow_datasets as tfds

from . import path


def get_endpoint_index(spec, size):
    if not spec:
        idx = None
    elif spec[-1] == '%':
        frac = float(spec[:-1]) / 100
        idx = round(frac*size)
    else:
        idx = int(spec)
    return idx


def get_dataset_size(config):
    config = copy.deepcopy(config)
    if 'tfds.load' in config:
        name_without_vrs = config['tfds.load']['name'].split(':')[0]
        repo_root = path.get_repo_root()
        data_dir = os.path.join(repo_root, 'datasets', name_without_vrs)
        if 'data_dir' not in config['tfds.load']:
            config['tfds.load']['data_dir'] = data_dir
        _, info = tfds.load(**config['tfds.load'], with_info=True)
        split_spec = config['tfds.load']['split']
        if '[' in split_spec:
            split_name, split_endpoints = split_spec.split('[')
            split_endpoints = split_endpoints[:-1].split(':')
            split_size = info.splits[split_name].num_examples
            split_endpoints = [get_endpoint_index(ep, split_size) for ep in split_endpoints]
            if split_endpoints[0] is None:
                split_endpoints[0] = 0
            if split_endpoints[1] is None:
                split_endpoints[1] = split_size
            size = split_endpoints[1] - split_endpoints[0]
        else:
            size = info.splits[split_spec].num_examples
    else:
        raise ValueError("Cannot compute dataset size for config:\n{}".format(config))
    return size
