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

options = vars(parser.parse_args())


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from counterfactual import generate_trajectory
from networks import build_networks, save_networks, get_optimizers
from options import save_options, load_options, get_current_epoch


options = load_options(options)

dataloader = CustomDataloader(**options)
networks = build_networks(dataloader.num_classes, **options)

start_epoch = get_current_epoch(options['result_dir'])
print("Loaded model at epoch {}".format(start_epoch))

# Generate a counterfactual
generate_trajectory(networks, dataloader, **options)
