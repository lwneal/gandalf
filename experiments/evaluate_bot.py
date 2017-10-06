#!/usr/bin/env python
"""
Usage:

    evaluate_bot.py
    
Finds a result_dir that hasn't been fully evaluated.
Runs evaluate_classifier.py on that result_dir.
"""
import os
import random
import json
from docopt import docopt

RESULTS_DIR = os.path.expanduser('~/results')


def get_eval(result_dir, epoch, fold='test'):
    filename = 'eval_epoch_{:04d}.json'.format(epoch)
    filename = os.path.join(result_dir, filename)
    if not os.path.exists(filename):
        print("No eval exists for result {} epoch {:04d}".format(result_dir, epoch))
        return None
    evals = json.load(open(filename))
    if fold in evals:
        return evals[fold]
    print("Result {} epoch {} only includes folds: {}".format(
        result_dir, epoch, evals.keys()))
    return None


def run_eval(result_dir, epoch, fold='test'):
    # TODO: security lol
    cmd = 'python experiments/evaluate_classifier.py --result_dir {} --epoch {} --fold {}'.format(
            result_dir, epoch, fold)
    os.system(cmd)


def run_plots(result_dir):
    cmd = 'python experiments/plot_graphs.py --result_dir {}'.format(result_dir)
    os.system(cmd)


def epoch_from_filename(filename):
    numbers = filename.split('epoch_')[-1].rstrip('.pth')
    return int(numbers)


def main():
    result_dir_listing = os.listdir(RESULTS_DIR)
    random.shuffle(result_dir_listing)
    for r in result_dir_listing:
        result_dir = os.path.join(RESULTS_DIR, r)
        if not os.path.isdir(result_dir):
            print("Skipping non-directory file {}".format(result_dir))
            continue
        files = os.listdir(result_dir)

        print("Reading checkpoints in {}".format(result_dir))
        encoder_checkpoints = [f for f in files if f.startswith('generator') and f.endswith('.pth')]
        encoder_checkpoints.sort()

        for fold in ['test']:
            print("Checking evaluations for fold {}".format(fold))
            for checkpoint_file in encoder_checkpoints:
                epoch = epoch_from_filename(checkpoint_file)
                if get_eval(result_dir, epoch, fold) is None:
                    print("Evaluation has not been done for {} epoch {} fold {}".format(
                        result_dir, epoch, fold))
                    run_eval(result_dir, epoch, fold)
                print("Finished running evaulation {} {} {}".format(result_dir, epoch, fold))
        print("All evaluations are completed for {}".format(r))
        run_plots(result_dir)


if __name__ == '__main__':
    args = docopt(__doc__)
    main()
