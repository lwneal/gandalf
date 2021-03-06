import os
import subprocess
import json
import argparse
from pprint import pprint


def save_options(options):
    # Include the version of the code that saved the options
    # (in case the meaning of an option changes in the future)
    if 'version' not in options:
        options['version'] = get_code_version()
    if not os.path.exists(options['result_dir']):
        print("Creating result directory {}".format(options['result_dir']))
        os.makedirs(options['result_dir'])

    filename = os.path.join(options['result_dir'], 'params.json')
    with open(filename, 'w') as fp:
        print("Saving options to {}".format(filename))
        to_save = options.copy()
        # Do not save result_dir; always read it from the command line
        del to_save['result_dir']
        json.dump(to_save, fp, indent=2, sort_keys=True)


def load_options(options):
    print("Resuming existing experiment at {} with options:".format(options['result_dir']))
    old_opts = json.load(open(os.path.join(options['result_dir'], 'params.json')))

    options.update(old_opts)
    options['result_dir'] = os.path.expanduser(options['result_dir'])

    pprint(options)
    return options


def get_current_epoch(result_dir):
    filenames = os.listdir(os.path.expanduser(result_dir))
    model_filenames = [f for f in filenames if f.endswith('.pth')]
    if not model_filenames:
        return 0
    def filename_to_epoch(filename):
        tokens = filename.rstrip('.pth').split('_')
        try:
            return int(tokens[-1])
        except ValueError:
            return 0
    return max(filename_to_epoch(f) for f in model_filenames)


def get_code_version():
    cwd = os.path.dirname(__file__)
    output = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd)
    return output.strip().decode('utf-8')
