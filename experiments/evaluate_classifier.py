#!/usr/bin/env python
import argparse
import json
import os
import sys

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')
parser.add_argument('--fold', default="test", help='Name of evaluation fold [default: test]')
parser.add_argument('--epoch', type=int, help='Epoch to evaluate (latest epoch if none chosen)')
parser.add_argument('--save_latent_vectors', default=False, help='Save Z in .npy format for later visualization')
parser.add_argument('--comparison_dataset', type=str, help='Dataset for off-manifold comparison')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks
from options import load_options, get_current_epoch
from evaluation import evaluate_classifier, save_evaluation

options = load_options(options)
if not options.get('epoch'):
    options['epoch'] = get_current_epoch(options['result_dir'])
options['random_horizontal_flip'] = False

dataloader = CustomDataloader(last_batch=True, shuffle=False, **options)

networks = build_networks(dataloader.num_classes, **options)

comparison_dataloader = None
if options['comparison_dataset']:
    comparison_options = options.copy()
    comparison_options['dataset'] = options['comparison_dataset']
    comparison_dataloader = CustomDataloader(last_batch=True, shuffle=False, **comparison_options)
    options['fold'] = 'openset_{}'.format(options['fold'])

new_results = evaluate_classifier(networks, dataloader, comparison_dataloader, **options)

save_evaluation(new_results, options['result_dir'], options['epoch'])

