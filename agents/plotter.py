#!/usr/bin/env python
"""
Usage:

    plotter.py
    
Searches for result_dir directories that could have charts
or visualizations generated, and enqueues jobs for them.
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


# TODO: better heuristic to decide whether this result_dir needs graphs plotted
def is_plotted(result_dir):
    filenames = os.listdir(os.path.join(RESULTS_DIR, result_dir))
    return 'plot_pca_test_epoch_0025.png' in filenames


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


def get_result_dirs():
    result_dirs = os.listdir(RESULTS_DIR)
    result_dirs = [os.path.join(RESULTS_DIR, r) for r in result_dirs]
    result_dirs = [r for r in result_dirs if is_valid_directory(r)]
    if len(result_dirs) == 0:
        print("Could not find any valid result_dir in {}".format(RESULTS_DIR))
    return result_dirs


def plot_all():
    result_dirs = get_result_dirs()
    # Collect all possible runnable jobs
    to_run = []
    for rd in result_dirs:
        if is_valid_directory(rd) and not is_plotted(rd):
            enqueue_plot_job(rd)


def enqueue_plot_job(result_dir):
    print("Plotting graphs for {}".format(result_dir))
    start_job('experiments/plot_graphs.py --result_dir {}'.format(
        os.path.join(RESULTS_DIR, result_dir)))


if __name__ == '__main__':
    args = docopt(__doc__)
    while True:
        time.sleep(10)
        if len(get_jobs()) > 0:
            continue
        print("Waking up at {}".format(int(time.time())))
        plot_all()
