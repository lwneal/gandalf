#!/usr/bin/env python
from __future__ import print_function
import json
import argparse
import os
import numpy as np
from pprint import pprint
import sys

def boolean(x):
    return not not x and x not in ['False', 'false', '0']

# Dataset (input) and result_dir (output) are always required
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

parser.add_argument('--classifier_epochs', type=int, default=20, help='Max number of training epochs [default: 20]')
parser.add_argument('--supervised', type=boolean, default=False, help='If True, include all training labels [default: False]')

options = vars(parser.parse_args())


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from training import train_active_learning
from evaluation import evaluate_classifier, save_evaluation
from networks import build_networks, get_optimizers
from options import save_options, load_options, get_current_epoch
from locking import acquire_lock, release_lock


loaded_options = load_options(options)

print("Loading full-sized dataset for evaluation...")
test_dataloader = CustomDataloader(fold='test', **options)

print("Loading networks (except classifier)...")
networks = build_networks(test_dataloader.num_classes, load_classifier=False, **options)
optimizers = get_optimizers(networks, **options)

print("Loading all available active-learning labels...")
active_label_dir = os.path.join(options['result_dir'], 'labels')
labels = []
for filename in os.listdir(active_label_dir):
    filename = os.path.join(active_label_dir, filename)
    info = json.load(open(filename))
    labels.append(info)

trajectory_dir = os.path.join(options['result_dir'], 'trajectories')
trajectory_filenames = os.listdir(trajectory_dir)
def get_trajectory_filename(trajectory_id):
    for filename in trajectory_filenames:
        if filename.endswith('.npy') and trajectory_id in filename:
            return filename
    return None

print("Loading Numpy arrays for label trajectories")

active_points = []
active_labels = []
complementary_points = []
complementary_labels = []
for label in labels:
    trajectory_filename = get_trajectory_filename(label['trajectory_id'])
    trajectory_filename = os.path.join(trajectory_dir, trajectory_filename)
    points = np.load(trajectory_filename)

    split_point = int(label['label_point'])

    start_class = test_dataloader.lab_conv.idx[label['start_class']]
    target_class = test_dataloader.lab_conv.idx[label['target_class']]

    # These are normal labels, eg "this picture is a cat"
    active_points.extend(np.squeeze(points[:split_point], axis=1))
    active_labels.extend([start_class] * split_point)

    # Complementary labels eg. "this picture is not a cat"
    # Might be a dog, might be a freight train, might be an adversarial image
    complementary_points.extend(np.squeeze(points[split_point:], axis=1))
    complementary_labels.extend([start_class] * (len(points) - split_point))

if options['supervised']:
    from torch.autograd import Variable
    netE = networks['encoder']
    train_dataloader = CustomDataloader(fold='train', **options)
    print("Appending {} batches of training set examples".format(len(train_dataloader)))
    for images, labels, _ in train_dataloader:
        z = netE(Variable(images))
        active_points.extend(z.data.cpu().numpy())
        active_labels.extend(labels.cpu().numpy())

print("Training classifier with {} positive and {} negative points".format(
    len(active_points), len(complementary_points)))
pprint({x: active_labels.count(x) for x in set(active_labels)})

active_points = np.array(active_points)
active_labels = np.array(active_labels)
complementary_points = np.array(complementary_points)
complementary_labels = np.array(complementary_labels)

best_score = 0
print("Training classifier using active-learning labels")
for classifier_epoch in range(options['classifier_epochs']):
    # Apply learning rate decay and train for one pseudo-epoch
    for optimizer in optimizers.values():
        optimizer.param_groups[0]['lr'] = .0001 * (.9 ** classifier_epoch)
    train_active_learning(networks, optimizers, active_points, active_labels, complementary_points, complementary_labels, **options)

    # Evaluate against the test set
    foldname = 'active_trajectories_{:06d}'.format(len(labels))
    if options['supervised']:
        foldname = 'active_supervised_trajectories_{:06d}'.format(len(labels))
    print("Evaluating {}".format(foldname))
    new_results = evaluate_classifier(networks, test_dataloader, verbose=False, fold=foldname, **options)

    print("Results from training with {} trajectories".format(len(labels)))
    pprint(new_results)
    new_score = new_results[foldname]['accuracy']
    if best_score < new_score:
        best_results = new_results
        best_score = new_score
        best_results[foldname]['best_classifier_epoch'] = classifier_epoch
save_evaluation(best_results, options['result_dir'], get_current_epoch(options['result_dir']))
