#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys

# Dataset (input) and result_dir (output) are always required
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

# Other options can change with every run
parser.add_argument('--batch_size', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--fold', type=str, default='test', help='Fold [default: test]')
parser.add_argument('--desired_class', type=int, help='Desired class number')
parser.add_argument('--mode', default="batch", help='One of: batch, active [default: batch]')

options = vars(parser.parse_args())


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
import counterfactual
from networks import build_networks, save_networks, get_optimizers
from options import save_options, load_options, get_current_epoch


options = load_options(options)

if options['mode'] == 'active':
    # Active Learning trajectories can only be single examples
    options['batch_size'] = 1
else:
    # Batch Labeling trajectories must be 4x4 grids
    options['batch_size'] = 16

dataloader = CustomDataloader(**options)
networks = build_networks(dataloader.num_classes, **options)

start_epoch = get_current_epoch(options['result_dir'])
print("Loaded model at epoch {}".format(start_epoch))

# Generate a counterfactual
if options['mode'] == 'batch':
    counterfactual.generate_trajectory_batch(networks, dataloader, **options)
elif options['mode'] == 'active':
    # TODO: Select the optimal start class and target class
    counterfactual.generate_trajectory_active(networks, dataloader, **options)
