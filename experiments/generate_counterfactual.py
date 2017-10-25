#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys

def is_true(x):
    return not not x and x.lower().startswith('t')

# Dataset (input) and result_dir (output) are always required
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

# Other options can change with every run
parser.add_argument('--batch_size', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--fold', type=str, default='test', help='Fold [default: test]')
parser.add_argument('--start_class', type=str, help='Name of starting class (random by default)')
parser.add_argument('--target_class', type=str, help='Name target class (random by default)')
parser.add_argument('--zero_attribute', type=str, help='Attribute to set to one')
parser.add_argument('--one_attribute', type=str, help='Attribute to set to zero')
parser.add_argument('--mode', default="batch", help='One of: batch, active, uncertainty [default: batch]')
parser.add_argument('--counterfactual_frame_count', default=60, type=int,
        help='Number of frames to output [default: 60]')
parser.add_argument('--classifier_name', type=str, default='active_learning_classifier',
        help='Name of the classifier to use [default: active_learning_classifier]')
parser.add_argument('--speed', type=float, default=.001, help='Learning rate for counterfactual descent [default: .001]')
parser.add_argument('--momentum_mu', type=float, default=.95, help='Momentum decay (zero for no momentum) [default: .95]')
parser.add_argument('--counterfactual_max_iters', type=int, default=1000, help='Maximum number of steps to take for CF trajectories [default: 1000]')
parser.add_argument('--write_jpgs', type=is_true, default=False, help='Write extra JPG files for visualization')
parser.add_argument('--start_epoch', type=int, help='Epoch to start from (defaults to most recent epoch)')

options = vars(parser.parse_args())


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
import counterfactual
from networks import build_networks, get_optimizers
from options import load_options, get_current_epoch


options = load_options(options)

if options['mode'] in ['active', 'uncertainty']:
    # Active Learning trajectories can only be single examples
    options['batch_size'] = 1
else:
    # Batch Labeling trajectories must be 4x4 grids
    options['batch_size'] = 64

dataloader = CustomDataloader(**options)

networks = build_networks(dataloader.num_classes, dataloader.num_attributes, **options)

start_epoch = options['start_epoch']
if start_epoch is None:
    start_epoch = get_current_epoch(options['result_dir'])
print("Loaded model at epoch {}".format(start_epoch))

if options['mode'] == 'batch':
    # This one is a good visualization and can be used for efficient labeling
    counterfactual.generate_trajectory_batch(networks, dataloader, **options)
elif options['mode'] == 'active':
    # This is the classic visualization for a single example
    counterfactual.generate_trajectory_active(networks, dataloader, **options)
elif options['mode'] == 'uncertainty':
    # Uncertainty Sampling used for Experiment 1. Requires --classifier_name
    counterfactual.generate_trajectory_active(networks, dataloader, strategy='uncertainty', **options)
