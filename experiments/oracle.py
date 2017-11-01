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
parser.add_argument('--oracle_pth', required=True, help="Oracle classifier network saved in .pth format")

options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from dataloader import CustomDataloader
from networks import build_networks
from options import load_options, get_current_epoch
from evaluation import evaluate_classifier, save_evaluation

options = load_options(options)

print("Loading networks from current result dir...")
dataloader = CustomDataloader(**options)
networks = build_networks(dataloader.num_classes, dataloader.num_attributes, **options)

print("Loading classifier weights from {}...".format(options['oracle_pth']))
networks['classifier'].load_state_dict(torch.load(options['oracle_pth']))
print("Loaded networks")

netC = networks['classifier']

import numpy as np
import torch
from torch.autograd import Variable

def classify(z, netC, dataloader):
    import torch
    from torch.autograd import Variable
    z = Variable(torch.FloatTensor(z)).cuda()
    pred = netC(z)
    pred_conf, pred_max = pred.max(1)
    pred_idx = pred_max.data.cpu().numpy()[0]
    return dataloader.lab_conv.labels[pred_idx]


print("Loading all available active-learning trajectories...")
trajectory_dir = os.path.join(options['result_dir'], 'trajectories')
labels_dir = os.path.join(options['result_dir'], 'labels')
if not os.path.exists(labels_dir):
    os.mkdir(labels_dir)
trajectory_filenames = os.listdir(trajectory_dir)
labeled_count = 0
for trajectory_filename in trajectory_filenames:
    if not trajectory_filename.endswith('npy'):
        continue
    trajectory_filename = os.path.join(trajectory_dir, trajectory_filename)
    start_class = trajectory_filename.split('-')[-2]
    target_class = trajectory_filename.split('-')[-1].rstrip('.npy')
    trajectory_id = trajectory_filename.split('-')[-3]

    points = np.load(trajectory_filename)
    labels = [classify(p, netC, dataloader) for p in points]
    for left_boundary in range(len(labels)):
        if labels[left_boundary] != start_class:
            break
    for right_boundary in range(len(labels), 0, -1):
        if labels[right_boundary - 1] != target_class:
            break

    label = {
            'start_label_point': left_boundary,
            'end_label_point': right_boundary,
            'trajectory_id': trajectory_id,
            'start_class': start_class,
            'target_class': target_class,
            'true_start_class': labels[0],
            'true_target_class': labels[-1],
    }
    label_filename = os.path.join(labels_dir, trajectory_id + '.json')
    with open(label_filename, 'w') as fp:
        fp.write(json.dumps(label, indent=2, sort_keys=True))
    labeled_count += 1

print("Oracle labeled {} trajectories".format(labeled_count))

