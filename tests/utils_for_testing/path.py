import os


def get_repo_root():
    start = os.path.abspath(__file__)
    while os.path.split(start)[1] != 'neuron-correlation':
        start = os.path.dirname(start)
    return start
