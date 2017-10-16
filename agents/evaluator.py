#!/usr/bin/env python
"""
Usage:

    evaluator.py
    
Searches for a result_dir that hasn't been fully evaluated.
Enqueues an appropriate evaluation job
"""
import os
import random
import json
from docopt import docopt
from subprocess import check_call


RESULTS_DIR = '/mnt/results'


def get_results(result_dir, epoch):
    filename = 'eval_epoch_{:04d}.json'.format(epoch)
    filename = os.path.join(result_dir, filename)
    if os.path.exists(filename):
        eval_folds = json.load(open(filename))
        return eval_folds
    return {}

def get_params(result_dir):
    filename = 'params.json'
    filename = os.path.join(RESULTS_DIR, result_dir, filename)
    return json.load(open(filename))


def get_dataset_name(result_dir):
    params = get_params(result_dir)
    dataset = params['dataset']
    return dataset.split('/')[-1].replace('.dataset', '')


def epoch_from_filename(filename):
    numbers = filename.split('epoch_')[-1].rstrip('.pth')
    return int(numbers)


def is_valid_directory(result_dir):
    result_dir = os.path.join(RESULTS_DIR, result_dir)
    if not os.path.exists(result_dir) or not os.path.isdir(result_dir):
        return False
    if 'params.json' not in os.listdir(result_dir):
        return False
    if 'robotno' in os.listdir(result_dir):
        print("Found robotno in {}, skipping".format(result_dir))
        return False
    return True


def get_epochs(result_dir):
    filenames = os.listdir(os.path.join(RESULTS_DIR, result_dir))
    pth_names = [f for f in filenames if f.endswith('.pth')]
    return sorted(list(set(epoch_from_filename(f) for f in pth_names)))


def choose_comparison_dataset(dataset):
    comparisons = {
        'cifar10': 'cub200_2011',
        'cub200_2011': 'cifar10',
        'cifar10-05': 'cifar10-69',
        'cifar10-69': 'cifar10-05',
        'cifar10-animals': 'cifar10-machines',
        'cifar10-machines': 'cifar10-animals',
        'mnist': 'emnist',
        'emnist': 'mnist',
        'mnist-05': 'mnist-69',
        'mnist-69': 'mnist-05',
        'oxford102': 'celeba',
        'celeba': 'oxford102',
    }
    return comparisons[dataset]


def get_eval_types(result_dir, epoch):
    # Evaluate classification and autoencoder performance
    folds = ['train', 'test']
    dataset_name = get_dataset_name(result_dir)

    # Evaluate open-set classification performance
    comparison_dataset = choose_comparison_dataset(dataset_name)
    folds.append('openset_test_comparison_{}'.format(comparison_dataset))

    # Evaluate semi-supervised classification performance
    folds.append('{}_example_count_001000'.format(dataset_name))
    return folds


def evaluate(result_dir, epoch, eval_type):
    results = get_results(result_dir, epoch)
    if eval_type not in results:
        print("TODO: evaluate {} epoch {} {}".format(result_dir, epoch, eval_type))


def get_result_dirs():
    result_dirs = os.listdir(RESULTS_DIR)
    result_dirs = [os.path.join(RESULTS_DIR, r) for r in result_dirs]
    result_dirs = [r for r in result_dirs if is_valid_directory(r)]
    if len(result_dirs) == 0:
        print("Could not find any valid result_dir in {}".format(RESULTS_DIR))
    return result_dirs


def main():
    result_dirs = get_result_dirs()
    for rd in result_dirs:
        epochs = get_epochs(rd)
        for epoch in epochs:
            eval_types = get_eval_types(rd, epoch)
            for eval_type in eval_types:
                evaluate(rd, epoch, eval_type)
    print("Finished checking {} result_dirs".format(len(result_dirs)))


if __name__ == '__main__':
    args = docopt(__doc__)
    main()
