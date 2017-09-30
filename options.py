import os
import json
import argparse
from pprint import pprint


def save_options(options):
    if not os.path.exists(options['result_dir']):
        print("Creating result directory {}".format(options['result_dir']))
        os.makedirs(options['result_dir'])
    filename = os.path.join(options['result_dir'], 'params.json')
    with open(filename, 'w') as fp:
        print("Saving options to {}".format(filename))
        json.dump(options, fp)


def load_options(options, core_options=None):
    print("Resuming existing experiment at {} with options:".format(options['result_dir']))
    old_opts = json.load(open(os.path.join(options['result_dir'], 'params.json')))
    options.update(old_opts)
    pprint(options)
    return options


def get_current_epoch(result_dir):
    filenames = os.listdir(os.path.expanduser(result_dir))
    model_filenames = [f for f in filenames if f.endswith('.pth')]
    if not model_filenames:
        return 0
    def filename_to_epoch(filename):
        tokens = filename.rstrip('.pth').split('_')
        return int(tokens[-1])
    return max(filename_to_epoch(f) for f in model_filenames)
