import sys
import json
import plotting
from imutil import show
import plotting


if __name__ == '__main__':
    filenames = [
        'avg_random.json',
        'avg_randomnearest.json',
        'avg_uncertaintynearest.json',
        'avg_uncertainty_random.json',
        'uncertainty_sampling_baseline.json'
    ]
    names = [
            'CF Random',
            'CF Random-Nearest',
            'CF Uncertainty Nearest',
            'CF Uncertainty Random',
            'Uncertainty Sampling Baseline']
    plotting.compare_multiple(filenames, names, 'comparison.png')
