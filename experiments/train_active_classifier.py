#!/usr/bin/env python
import time
import argparse
import json
import os
import sys
from pprint import pprint

def is_true(x):
    return not not x and x.lower().startswith('t')

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')
parser.add_argument('--classifier_name', type=str, default='active_learning_classifier',
        help='Name of the classifier to use [default: active_learning_classifier]')
parser.add_argument('--max_trajectories', type=int, help='Number of trajectories to train with (default: uses all available trajectories)')
parser.add_argument('--use_trajectories', type=is_true, default=True, help='If True, use trajectories, if False use only initial points')
parser.add_argument('--use_negative_labels', type=is_true, default=True, help='If False, ignore all negative labels')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks, get_optimizers, save_networks
from options import load_options, get_current_epoch
from evaluation import evaluate_classifier, save_evaluation
from training import train_active_learning

options = load_options(options)
current_epoch = get_current_epoch(options['result_dir'])
classifier_name = options['classifier_name']

# We start with a small set of ground-truth labels to seed the classifier: 1 label per class
dataloader = CustomDataloader(fold='train', **options)
# We can also test while training to get data
test_dataloader = CustomDataloader(fold='test', **options)

# Load the active learning classifier and the unsupervised encoder/generator
networks = build_networks(dataloader.num_classes, dataloader.num_attributes, epoch=current_epoch, 
        load_classifier=False, load_attributes=False, **options)
optimizers = get_optimizers(networks, **options)


print("Loading all available active-learning labels...")
active_label_dir = os.path.join(options['result_dir'], 'labels')
if not os.path.exists(active_label_dir):
    os.mkdir(active_label_dir)
labels = []
label_filenames = os.listdir(active_label_dir)
label_filenames.sort()
if options['max_trajectories']:
    print("Limiting labels to first {}".format(options['max_trajectories']))
    label_filenames = label_filenames[:options['max_trajectories']]


for filename in label_filenames:
    filename = os.path.join(active_label_dir, filename)
    info = json.load(open(filename))
    labels.append(info)
print("Loaded {} labels from {}".format(len(labels), active_label_dir))

import numpy as np
print("Loading all available active-learning trajectories...")
trajectory_dir = os.path.join(options['result_dir'], 'trajectories')
if not os.path.exists(trajectory_dir):
    os.mkdir(trajectory_dir)
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
    if options['use_trajectories']:
        # Decision boundary labeling: a set of points on either side
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
    else:
        # A standard active learning setup: only a single point
        true_start_class = dataloader.lab_conv.idx[label['true_start_class']]
        active_points.append(points[0])
        active_labels.append(true_start_class)

active_points = np.array(active_points)
active_labels = np.array(active_labels)
complementary_points = np.array(complementary_points)
complementary_labels = np.array(complementary_labels)
print("Loaded {} trajectories totalling {} positive points and {} negative points".format(
	len(labels), len(active_points), len(complementary_points)))

def augment_to_length(points, labels, required_len=2000):
    assert len(points) > 0
    assert len(points) == len(labels)
    augmented_points = np.copy(points)
    augmented_labels = np.copy(labels)
    while len(augmented_points) < required_len:
        x = min(required_len - len(augmented_points), len(points))
        augmented_points = np.concatenate([augmented_points, points[:x]])
        augmented_labels = np.concatenate([augmented_labels, labels[:x]])
    print("Augmented points up to length {}".format(len(active_points)))
    return augmented_points, augmented_labels

if len(active_points) < 2**12:
    print("Padding active label dataset for training stability")
    active_points, active_labels = augment_to_length(active_points, active_labels, required_len=2**12)

# Also add some set number of regular labels, like 1000
extra_points = []
extra_labels = []
netE = networks['encoder']
from torch.autograd import Variable
for (images, labels, _) in dataloader:
    extra_points.extend(netE(Variable(images)).data.cpu().numpy())
    extra_labels.extend(labels.cpu().numpy())
    if len(extra_points) > 1000:
        break
active_points = np.concatenate([active_points, extra_points])
active_labels = np.concatenate([active_labels, extra_labels])


best_score = 0
print("Re-training classifier {} using {} active-learning label points".format(
    classifier_name, len(active_points) + len(complementary_points)))

MAX_EPOCHS = 25
for classifier_epoch in range(MAX_EPOCHS):
    # Apply learning rate decay and train for one pseudo-epoch
    for optimizer in optimizers.values():
        optimizer.param_groups[0]['lr'] = .0001 * (.9 ** classifier_epoch)


    start_time = time.time()
    train_active_learning(networks, optimizers, active_points, active_labels, complementary_points, complementary_labels, **options)
    print("Ran train_active_learning in {:.3f}s".format(time.time() - start_time))

    # Evaluate against the test set
    foldname = 'active_trajectories_{:06d}'.format(len(labels))

    print("Evaluating {}".format(foldname))
    start_time = time.time()
    new_results = evaluate_classifier(networks, test_dataloader, verbose=False, fold=foldname, skip_reconstruction=True, **options)
    print("Ran evaluate_classifier in {:.3f}s".format(time.time() - start_time))

    print("Results from training with {} trajectories".format(len(labels)))
    pprint(new_results)
    new_score = new_results[foldname]['accuracy']
    if best_score < new_score:
        best_results = new_results
        best_score = new_score
        best_results[foldname]['best_classifier_epoch'] = classifier_epoch
    else:
        print("Overfit detected")
        #break
print("Finished re-training classifier, got test results:")
pprint(new_results)
print("Best results:")
pprint(best_results)

save_networks({classifier_name: networks[classifier_name]}, epoch=current_epoch, result_dir=options['result_dir'])

save_evaluation(best_results, options['result_dir'], get_current_epoch(options['result_dir']))
