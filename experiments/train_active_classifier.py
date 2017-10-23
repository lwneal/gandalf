#!/usr/bin/env python
import argparse
import json
import os
import sys

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')
parser.add_argument('--classifier_name', type=str, default='active_learning_classifier',
        help='Name of the classifier to use [default: active_learning_classifier]')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks, save_networks
from options import load_options, get_current_epoch
from evaluation import evaluate_classifier, save_evaluation

options = load_options(options)
current_epoch = get_current_epoch(options['result_dir'])

# We start with a small set of ground-truth labels to seed the classifier: 1 label per class
dataloader = CustomDataloader(fold='train', **options)

# Load the active learning classifier and the unsupervised encoder/generator
networks = build_networks(dataloader.num_classes, dataloader.num_attributes, epoch=current_epoch, **options)


print("Loading all available active-learning labels...")
active_label_dir = os.path.join(options['result_dir'], 'labels')
labels = []
for filename in os.listdir(active_label_dir):
    filename = os.path.join(active_label_dir, filename)
    info = json.load(open(filename))
    labels.append(info)
print("Loaded {} labels from {}".format(len(labels), active_label_dir))

import numpy as np
print("Loading all available active-learning trajectories...")
trajectory_dir = os.path.join(options['result_dir'], 'trajectories')
trajectory_filenames = os.listdir(trajectory_dir)
def get_trajectory_filename(trajectory_id):
    for filename in trajectory_filenames:
        if filename.endswith('.npy') and trajectory_id in filename:
            return filename
    return None
active_points = []
active_labels = []
complementary_points = []
complementary_labels = []
for label in labels:
    trajectory_filename = get_trajectory_filename(label['trajectory_id'])
    trajectory_filename = os.path.join(trajectory_dir, trajectory_filename)
    points = np.load(trajectory_filename)
    split_point = int(label['label_point'])
    start_class = dataloader.lab_conv.idx[label['start_class']]
    target_class = dataloader.lab_conv.idx[label['target_class']]
    # These are normal labels, eg "this picture is a cat"
    active_points.extend(np.squeeze(points[:split_point], axis=1))
    active_labels.extend([start_class] * split_point)
    # Complementary labels eg. "this picture is not a cat"
    # Might be a dog, might be a freight train, might be an adversarial image
    complementary_points.extend(np.squeeze(points[split_point:], axis=1))
    complementary_labels.extend([start_class] * (len(points) - split_point))
active_points = np.array(active_points)
active_labels = np.array(active_labels)
complementary_points = np.array(complementary_points)
complementary_labels = np.array(complementary_labels)
print("Loaded {} trajectories totalling {} positive points and {} negative points".format(
	len(labels), len(active_points), len(complementary_points)))


print("# TODO: Use those labeled points to re-train the classifier")

classifier_name = options['classifier_name']
save_networks({classifier_name: networks[classifier_name]}, epoch=current_epoch, result_dir=options['result_dir'])

