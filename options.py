import os
import json
import argparse
from pprint import pprint


def save_options(options, core_options):
    print("Beginning a new experiment at {}".format(options['result_dir']))
    for opt_name in core_options:
        if not options.get(opt_name):
            print("Error: required option --{} was not specified".format(opt_name))
            raise ValueError('Required option "{}" not specified'.format(opt_name))
    print("Creating directory {}".format(options['result_dir']))
    if not os.path.exists(options['result_dir']):
        os.makedirs(options['result_dir'])
    with open(os.path.join(options['result_dir'], 'params.json'), 'w') as fp:
        json.dump(options, fp)


def load_options(options, core_options):
    print("Resuming existing experiment at {}".format(options['result_dir']))
    old_opts = json.load(open(os.path.join(options['result_dir'], 'params.json')))
    pprint(old_opts)
    for opt_name in core_options:
        options[opt_name] = old_opts[opt_name]
        print("Setting {} to {}".format(opt_name, options[opt_name]))
    return options


def get_current_epoch(result_dir):
    filenames = os.listdir(os.path.expanduser(result_dir))
    model_filenames = [f for f in filenames if f.endswith('.pth')]
    def filename_to_epoch(filename):
        tokens = filename.rstrip('.pth').split('_')
        return int(tokens[-1])
    return max(filename_to_epoch(f) for f in model_filenames) + 1
