#!/usr/bin/env python
import argparse
import json
import os
import sys
from pprint import pprint

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')
parser.add_argument('--fold', default="test", help='Name of evaluation fold [default: test]')
parser.add_argument('--save_latent_vectors', default=False, help='Save Z in .npy format for later visualization')
parser.add_argument('--example_parameter', type=int, default=42, help='Example parameter that will be in **options')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import examples
from options import load_options, get_current_epoch
from dataloader import CustomDataloader
from networks import build_networks
from evaluation import evaluate_classifier
from locking import acquire_lock, release_lock

print("Loading any available saved options from {}/params.json".format(options['result_dir']))
options = load_options(options)

print("Switching to the most recent version of the network saved in {}".format(options['result_dir']))
options['epoch'] = get_current_epoch(options['result_dir'])

print("Loading dataset from file {}".format(options['dataset']))
dataloader = CustomDataloader(last_batch=True, shuffle=False, **options)

print("Loading neural network weights...")
nets = build_networks(dataloader.num_classes, **options)


examples.run_example_code(nets, dataloader, **options)


print("Evaluating the accuracy of the classifier on the {} fold".format(options['fold']))
new_results = evaluate_classifier(nets, dataloader, verbose=False, **options)

print("Results from evaluate_classifier:")
pprint(new_results)

acquire_lock(options['result_dir'])
try:
    print("Saving results in {}".format(options['result_dir']))
    filename = os.path.join(options['result_dir'], 'example_results.json')
    with open(filename, 'w') as fp:
        fp.write(json.dumps(new_results, indent=2))
finally:
    release_lock(options['result_dir'])
