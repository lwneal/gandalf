#!/usr/bin/env python
from __future__ import print_function
import json
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks
from options import load_options, get_current_epoch
from evaluation import evaluate_classifier

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')
parser.add_argument('--fold', default="test", help='Name of evaluation fold [default: test]')
parser.add_argument('--epoch', type=int, help='Epoch to evaluate (latest epoch if none chosen)')

options = vars(parser.parse_args())

options = load_options(options)

dataloader = CustomDataloader(last_batch=True, shuffle=False, **options)
networks = build_networks(dataloader.num_classes, **options)
epoch = get_current_epoch(options['result_dir'])

new_results = evaluate_classifier(networks, dataloader, **options)

filename = 'eval_epoch_{:04d}.json'.format(epoch)
filename = os.path.join(options['result_dir'], filename)
filename = os.path.expanduser(filename)

old_results = {}

if os.path.exists(filename):
    old_results = json.load(open(filename))

old_results.update(new_results)
with open(filename, 'w') as fp:
    json.dump(old_results, fp)
