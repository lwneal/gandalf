#!/usr/bin/env python
"""
Usage:

    evaluator.py
    
Searches for evaluations that have not been completed.
Enqueues jobs to complete them.
"""
import os
import random
import json
import rq
import redis
import pickle
import time
from docopt import docopt
from subprocess import check_call


RESULTS_DIR = '/mnt/results'
DATA_DIR = '/mnt/data'


conn = redis.Redis()


def should_enqueue_low_priority_jobs():
    # TODO: Some heuristic here
    return False


def is_last_epoch(result_dir, epoch):
    params = get_params(result_dir)
    return epoch == params['epochs']


def get_jobs():
    q = rq.Queue(connection=conn)
    jobs = []
    for job_id in q.job_ids:
        job = q.fetch_job(job_id)
        kwargs = pickle.loads(job.data)[-1]
        jobs.append({
            'enqueued_at': str(job.enqueued_at),
        })
    return jobs


def start_job(command):
    kwargs = {
        'directory': '~/gandalf',
        'command': command,
    }
    q = rq.Queue(connection=conn)
    # Timeout: 7 days
    timeout = 7 * 24 * 60 * 60
    q.enqueue('gromdar.client.jobs.run_experiment', kwargs=kwargs, timeout=timeout)


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
    dirs = os.listdir(result_dir)
    if 'robotno' in dirs or 'norobot' in dirs:
        print("Found robotno in {}, skipping".format(result_dir))
        return False
    return True


def get_epochs(result_dir):
    filenames = os.listdir(os.path.join(RESULTS_DIR, result_dir))
    pth_names = [f for f in filenames if f.endswith('.pth')]
    return sorted(list(set(epoch_from_filename(f) for f in pth_names)))


def comparison_dataset_for(dataset):
    dataset = dataset.lower()
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
        'svhn': 'cifar10',
        'cifar10': 'svhn',
    }
    return comparisons.get(dataset, 'cifar10')


def get_eval_types(result_dir, epoch):
    # Always run the fast test set eval
    folds = ['test']
    dataset_name = get_dataset_name(result_dir)

    # Optionally run the other, slower evals
    if is_last_epoch(result_dir, epoch) or should_enqueue_low_priority_jobs():
        # Evaluate performance on the training set
        folds.append('train')

        # Evaluate open-set classification performance
        comparison_dataset = comparison_dataset_for(dataset_name)
        folds.append('openset_test_comparison_{}'.format(comparison_dataset))

        # Evaluate semi-supervised classification performance
        folds.append('{}_example_count_001000'.format(dataset_name))
    else:
        print("Skipping low-priority evals for {} {}".format(result_dir, epoch))
    return folds


def cmd_for_eval(result_dir, epoch, eval_type):
    dataset_name = get_dataset_name(result_dir)
    if eval_type.startswith('openset_test'):
        other_dataset = '{}.dataset'.format(comparison_dataset_for(dataset_name))
        other_dataset = os.path.join(DATA_DIR, other_dataset)
        return ['experiments/evaluate_openset.py',
                '--epoch', str(epoch),
                '--result_dir', os.path.join(RESULTS_DIR, result_dir),
                '--comparison_dataset', other_dataset]
    elif eval_type.endswith('example_count_001000'):
        # TODO: Semisupervised only supports 1k examples for now
        return ['experiments/evaluate_semisupervised.py',
                '--evaluation_epoch', str(epoch),
                '--result_dir', os.path.join(RESULTS_DIR, result_dir),
                '--evaluation_epoch', str(epoch)]
    else:
        # Otherwise eval_type is eg. 'train' or 'test'
        return ['experiments/evaluate_classifier.py',
                '--epoch', str(epoch),
                '--result_dir', os.path.join(RESULTS_DIR, result_dir),
                '--fold', eval_type]


def get_result_dirs():
    result_dirs = os.listdir(RESULTS_DIR)
    result_dirs = [os.path.join(RESULTS_DIR, r) for r in result_dirs]
    result_dirs = [r for r in result_dirs if is_valid_directory(r)]
    if len(result_dirs) == 0:
        print("Could not find any valid result_dir in {}".format(RESULTS_DIR))
    return result_dirs


def evaluate_all():
    result_dirs = get_result_dirs()
    # Collect all possible runnable eval jobs
    to_run = []
    for rd in result_dirs:
        try:
            epochs = get_epochs(rd)
            for epoch in epochs:
                eval_types = get_eval_types(rd, epoch)
                for eval_type in eval_types:
                    to_run.append((rd, epoch, eval_type))
        except Exception as e:
            print("Warning: skipping bad result_dir {}".format(rd))
            print(e)
            continue

    # Run the ones that haven't been run
    random.shuffle(to_run)
    for (rd, epoch, eval_type) in to_run:
        results = get_results(rd, epoch)
        if eval_type not in results.keys():
            print("{}: {} not in {}".format(rd, eval_type, results.keys()))
            cmd = cmd_for_eval(rd, epoch, eval_type)
            start_job(' '.join(cmd))
    print("Finished checking {} result_dirs".format(len(result_dirs)))


if __name__ == '__main__':
    args = docopt(__doc__)
    while True:
        if len(get_jobs()) > 0:
            time.sleep(10)
            continue
        print("Waking up at {}".format(int(time.time())))
        evaluate_all()
        time.sleep(1)
