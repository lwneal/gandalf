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
parser.add_argument('--epoch', type=int, help='Epoch to evaluate (latest epoch if none chosen)')
parser.add_argument('--comparison_dataset', type=str, help='Dataset for off-manifold comparison')
parser.add_argument('--classifier_name', type=str, help='Name of classifier to use (eg active_learning_classifier)')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks
from options import load_options, get_current_epoch
from evaluation import evaluate_openset, save_evaluation

options = load_options(options)
if not options.get('epoch'):
    options['epoch'] = get_current_epoch(options['result_dir'])
options['random_horizontal_flip'] = False

dataloader_on = CustomDataloader(last_batch=True, shuffle=False, **options)

off_manifold_options = options.copy()
off_manifold_options['dataset'] = options['comparison_dataset']
dataloader_off = CustomDataloader(last_batch=True, shuffle=False, **off_manifold_options)

networks = build_networks(dataloader_on.num_classes, **options)

statistics = evaluate_openset(networks, dataloader_on, dataloader_off, **options)
pprint(statistics)

label_count = 0
label_path = os.path.join(options['result_dir'], 'labels')
if os.path.exists(label_path):
    label_count = len(os.listdir(label_path))
fold_name = 'openset_{}_comparison_{}_{:06d}'.format(options['fold'], dataloader_off.dsf.name, label_count)

results = {fold_name: statistics}
print("Saving evaluation statistics:")
pprint(results)
save_evaluation(results, options['result_dir'], options['epoch'])
