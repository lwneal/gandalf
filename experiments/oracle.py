#!/usr/bin/env python
# The Oracle reads .npy trajectory files and outputs .json labels, just like
# the labeling UI does, except no human is required
import argparse
import json
import os
import sys

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Directory to operate on (not the oracle)')
parser.add_argument('--oracle_result_dir', required=True, help="Load oracle from classifier.py in this result_dir")

options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks
from options import load_options, get_current_epoch
from evaluation import evaluate_classifier, save_evaluation

oracle_options = options.copy()

options = load_options(options)
oracle_options = load_options(oracle_options)

print("Loading networks from current result dir...")
dataloader = CustomDataloader(**options)
networks = build_networks(dataloader.num_classes, dataloader.num_attributes, **options)

print("Loading networks from oracle result dir...")
oracle_dataloader = CustomDataloader(**oracle_options)
oracle_networks = build_networks(oracle_dataloader.num_classes, oracle_dataloader.num_attributes, **oracle_options)

print("Loaded networks")
print("TODO: Use the oracle's classifier to get a classification for every point in every trajectory")

sourceG = networks['generator']
oracleE = oracle_networks['encoder']
oracleC = oracle_networks['classifier']

import numpy as np
import torch
from torch.autograd import Variable

print("Loading all available active-learning trajectories...")
trajectory_dir = os.path.join(options['result_dir'], 'trajectories')
labels_dir = os.path.join(options['result_dir'], 'labels')
trajectory_filenames = os.listdir(trajectory_dir)
trajectories = []
for trajectory_filename in trajectory_filenames:
    if not trajectory_filename.endswith('npy'):
        continue
    trajectory_filename = os.path.join(trajectory_dir, trajectory_filename)
    points = np.load(trajectory_filename)
    trajectories.append(points)
    prev_pred_class = None
    for i, p in enumerate(points):
        z = Variable(torch.FloatTensor(p)).cuda()
        pred = oracleC(oracleE(sourceG(z)))
        pred_conf, pred_max = pred.max(1)
        pred_class = pred_max.data.cpu().numpy()
        if i > 0 and pred_class != prev_pred_class:
            break
        prev_pred_class = pred_class
    print("Got trajectory {}".format(trajectory_filename))
    trajectory_id = trajectory_filename.split('-')[-3]
    label = {
            'label_point': i,
            'trajectory_id': trajectory_id,
            'start_class': trajectory_filename.split('-')[-2],
            'target_class': trajectory_filename.split('-')[-1].rstrip('.npy'),
    }
    label_filename = os.path.join(labels_dir, trajectory_id + '.json')
    with open(label_filename, 'w') as fp:
        fp.write(json.dumps(label, indent=2))
print("Loaded {} trajectories".format(len(trajectories)))

