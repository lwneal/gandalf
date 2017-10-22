#!/usr/bin/env python
"""
Usage:

    optimizer.py [eval_type metric_name]
    
Adjusts available parameters to maximize the given metric.
Defaults to "test accuracy"

eval_type: One of train, test, openset, etc.
metric_name: One of accuracy, MSE, auc, etc.
"""
import os
import uuid
import sys
import random
import json
import rq
import redis
import pickle
import time
import numpy as np
from docopt import docopt
from subprocess import check_call
from pprint import pprint


RESULTS_DIR = '/mnt/results'
DATA_DIR = '/mnt/data'
PATIENCE_SEC = 10

conn = redis.Redis()


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


def run_new_experiment(params):
    result_dir = params['result_dir']
    os.mkdir(result_dir)
    with open(os.path.join(result_dir, 'params.json'), 'w') as fp:
        fp.write(json.dumps(params, indent=2))
    cmd = "experiments/train_aac.py --result_dir {}".format(result_dir)
    start_job(cmd)


def get_result_dirs():
    return [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]


def get_results(result_dir, epoch):
    filename = 'eval_epoch_{:04d}.json'.format(epoch)
    filename = os.path.join(RESULTS_DIR, result_dir, filename)
    if os.path.exists(filename):
        eval_folds = json.load(open(filename))
        return eval_folds
    return {}


def get_params(result_dir):
    filename = 'params.json'
    filename = os.path.join(RESULTS_DIR, result_dir, filename)
    return json.load(open(filename))


def get_editable_params(result_dir):
    # TODO: Allow optimization of discrete parameters
    editable_params = {}
    params = get_params(result_dir)
    for name in params:
        if type(params[name]) is float or name in ['latent_size', 'epochs']:
            editable_params[name] = params[name]
    return editable_params


def perturb_editable_params(editable_params):
    # Select one parameter at random and move it up or down
    name = random.choice(list(editable_params.keys()))
    multiplier = 1 + np.random.uniform(-.1, .1)
    param_type = type(editable_params[name])
    new_val = param_type(editable_params[name] * multiplier)
    print("Setting {} to {}".format(name, new_val))
    editable_params[name] = new_val
    return editable_params


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


def get_all_info(fold, metric, dataset):
    info = []
    for result_dir in get_result_dirs():
        if get_dataset_name(result_dir) != dataset:
            continue
        # Evaluate the most recent epoch
        epochs = get_epochs(result_dir)
        if not epochs:
            continue
        epoch = epochs[-1]
        results = get_results(result_dir, epoch)
        if not results:
            continue
        if fold not in results:
            continue
        # HACK: unsupervised only
        if get_params(result_dir).get('supervised_encoder') != False:
            print("Skipping supervised {}".format(result_dir))
            continue
        info.append((result_dir, results[fold][metric]))
    info.sort(key=lambda x: x[1])
    return info


def start_new_job():
    dataset = 'emnist'
    fold = 'test'
    metric = 'accuracy'
    if len(sys.argv) > 1:
        fold = sys.argv[1]
    if len(sys.argv) > 2:
        metric = sys.argv[2]

    infos = get_all_info(fold=fold, metric=metric, dataset=dataset)
    if not infos:
        print("Error: No runs found for dataset {}".format(dataset))
        return

    print('{:<64} {:>8}'.format("Experiment Name", fold + '_' + metric))
    for (name, metric_val) in infos:
        print('{:<64} {:>8.4f}'.format(name, metric_val))

    best_result_dir, best_score = infos[-1]
    best_params = get_params(best_result_dir)

    print("The best {} run so far was {} with {} = {}".format(
        dataset, best_result_dir, metric, best_score))

    print("Parameters for best run:")
    pprint(best_params)

    print("Parameters for next run:")
    new_params = best_params.copy()

    perturbed = perturb_editable_params(get_editable_params(best_result_dir))
    new_params.update(perturbed)

    random_hex = uuid.uuid4().hex[-8:]
    image_size = new_params['image_size']
    new_result_dir = '{0}_{1}x{1}_{2}'.format(dataset, image_size, random_hex)
    new_params['result_dir'] = os.path.join(RESULTS_DIR, new_result_dir)
    pprint(new_params)

    run_new_experiment(new_params)
            

if __name__ == '__main__':
    args = docopt(__doc__)
    while True:
        for _ in range(PATIENCE_SEC):
            time.sleep(1)
            if len(get_jobs()) > 0:
                print("Jobs are on the queue. Sleeping.")
                time.sleep(10)
                break
        else:
            print("No jobs were enqueued for {} seconds. Starting the next experiment".format(PATIENCE_SEC))
            start_new_job()
