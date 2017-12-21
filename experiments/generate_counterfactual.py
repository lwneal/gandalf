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
parser.add_argument('--fold', type=str, default='train', help='Fold [default: train]')
parser.add_argument('--start_class', type=str, help='Name of starting class (random by default)')
parser.add_argument('--target_class', type=str, help='Name target class (random by default)')
parser.add_argument('--counterfactual_frame_count', default=60, type=int,
        help='Number of frames to output [default: 60]')
parser.add_argument('--classifier_name', type=str, default='active_learning_classifier',
        help='Name of the classifier to use [default: active_learning_classifier]')
parser.add_argument('--speed', type=float, default=.0001, help='Learning rate for counterfactual descent [default: .0001]')
parser.add_argument('--momentum_mu', type=float, default=.99, help='Momentum decay (zero for no momentum) [default: .99]')
parser.add_argument('--counterfactual_max_iters', type=int, default=1000, help='Maximum number of steps to take for CF trajectories [default: 1000]')
parser.add_argument('--start_epoch', type=int, help='Epoch to start from (defaults to most recent epoch)')
parser.add_argument('--count', type=int, default=1, help='Number of counterfactuals to generate')
parser.add_argument('--strategy', type=str, default='uncertainty', help='One of: random, uncertainty [default: uncertainty]')
parser.add_argument('--ground_truth', type=is_true, default=False, help='If True, output ground-truth training examples')

options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
import counterfactual
from networks import build_networks
from options import load_options


options = load_options(options)


dataloader = CustomDataloader(**options)

# Generate K x K grids of examples (to visualize K classes)
options['batch_size'] = dataloader.num_classes

networks = build_networks(dataloader.num_classes, **options)

for i in range(options['count']):
    if options['ground_truth']:
        counterfactual.generate_grid(networks, dataloader, **options)
    else:
        counterfactual.generate_comparison(networks, dataloader, **options)
