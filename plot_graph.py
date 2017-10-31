import sys
import json
import plotting
from imutil import show
import plotting


if __name__ == '__main__':
    filenames = [
        'uncertainty_sampling-monday.json',
        'random-monday.json',
        'random_nearest-monday.json',
        'uncertainty_nearest-monday.json',
        'uncertainty_random-monday.json',
        'certainty_random-monday.json',
        'certainty_furthest-monday.json',
    ]
    names = [
        'Uncertainty Sampling',
        'CF Random-Random',
        'CF Random-Nearest',
        'CF Uncertainty-Nearest',
        'CF Uncertainty-Random',
        'CF Certainty-Random',
        'CF Certainty-Furthest',
    ]
    plotting.compare_multiple(filenames, names, 'comparison.png')
