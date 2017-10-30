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
parser.add_argument('--init_label_count', type=int, help='Number of labels to initialize with. [default: 1000]', default=1000)
parser.add_argument('--query_count', type=int, help='Number of active learning queries to apply')
parser.add_argument('--experiment_type', type=str, help='One of: semisupervised, uncertainty_sampling, counterfactual')

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
result_dir = options['result_dir']

# We start with a small set of ground-truth labels to seed the classifier: 1 label per class
train_dataloader = CustomDataloader(fold='train', shuffle=False, **options)
# We can also test while training to get data
test_dataloader = CustomDataloader(fold='test', **options)

# Load the active learning classifier and the unsupervised encoder/generator
networks = build_networks(train_dataloader.num_classes, train_dataloader.num_attributes, epoch=current_epoch, 
        load_classifier=False, load_attributes=False, **options)
optimizers = get_optimizers(networks, **options)


def load_labels(active_label_dir, label_count):
    if not os.path.exists(active_label_dir):
        os.mkdir(active_label_dir)
    labels = []
    label_filenames = os.listdir(active_label_dir)
    label_filenames.sort()
    if label_count is not None:
        print("Limiting labels to first {}".format(label_count))
        label_filenames = label_filenames[:label_count]
    for filename in label_filenames:
        filename = os.path.join(active_label_dir, filename)
        info = json.load(open(filename))
        labels.append(info)
    return labels

def get_trajectory_filename(result_dir, trajectory_id):
    trajectory_dir = os.path.join(result_dir, 'trajectories')
    if not os.path.exists(trajectory_dir):
        os.mkdir(trajectory_dir)
    trajectory_filenames = os.listdir(trajectory_dir)
    for filename in trajectory_filenames:
        if filename.endswith('.npy') and trajectory_id in filename:
            return os.path.join(trajectory_dir, filename)
    return None


def load_active_learning_trajectories(labels, train_dataloader, margin):
    active_points = []
    active_labels = []
    complementary_points = []
    complementary_labels = []
    for label in labels:
        trajectory_filename = get_trajectory_filename(result_dir, label['trajectory_id'])
        points = np.load(trajectory_filename)
        # Decision boundary labeling: a set of points on either side
        split_point = int(label['label_point'])
        start_class = train_dataloader.lab_conv.idx[label['start_class']]
        target_class = train_dataloader.lab_conv.idx[label['target_class']]

        # These are normal labels, eg "this picture is a cat"
        # If margin = 2 then we throw away 2 points on either side of the boundary
        margin = 1
        left = max(0, split_point - margin)
        right = min(split_point + margin, len(points) - 1)
        active_points.extend(np.squeeze(points[:left], axis=1))
        active_labels.extend([start_class] * left)
        # Complementary labels eg. "this picture is not a cat"
        # Might be a dog, might be a freight train, might be an adversarial image
        complementary_points.extend(np.squeeze(points[right:], axis=1))
        complementary_labels.extend([start_class] * (len(points) - right))
    active_points = np.expand_dims(np.array(active_points), axis=1)
    complementary_points = np.expand_dims(np.array(complementary_points), axis=1)
    return active_points, np.array(active_labels), complementary_points, np.array(complementary_labels)


def load_active_learning_points(labels, train_dataloader):
    active_points = []
    active_labels = []
    for label in labels:
        trajectory_filename = get_trajectory_filename(result_dir, label['trajectory_id'])
        points = np.load(trajectory_filename)
        # A standard active learning setup: only a single point
        true_start_class = train_dataloader.lab_conv.idx[label['true_start_class']]
        active_points.append(points[0])
        active_labels.append(true_start_class)
    return np.array(active_points), np.array(active_labels)


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



print("Loading all available active-learning labels...")
active_label_dir = os.path.join(options['result_dir'], 'labels')
labels = load_labels(active_label_dir, label_count=options['query_count'])
foldname = 'active_trajectories_{:06d}'.format(len(labels))
print("Loaded {} labels from {}".format(len(labels), active_label_dir))

import numpy as np

if options['experiment_type'] == 'counterfactual':
    active_points, active_labels, complementary_points, complementary_labels = load_active_learning_trajectories(labels, train_dataloader, margin=2)
elif options['experiment_type'] == 'uncertainty_sampling':
    active_points, active_labels = load_active_learning_points(labels, train_dataloader)
    complementary_points = np.array([])
    complementary_labels = np.array([])
elif options['experiment_type'] == 'semisupervised':
    active_points, active_labels = load_active_learning_points(labels, train_dataloader)
    complementary_points = np.array([])
    complementary_labels = np.array([])
else:
    raise ValueError("Unknown experiment_type")
print("Loaded {} trajectories totalling {} positive points and {} negative points".format(
	len(labels), len(active_points), len(complementary_points)))


# Every run starts with a set of 1k labels
INIT_LABELS = options['init_label_count']
extra_points = []
extra_labels = []
netE = networks['encoder']
from torch.autograd import Variable
for (images, labels, _) in train_dataloader:
    extra_points.extend(netE(Variable(images)).data.cpu().numpy())
    extra_labels.extend(labels.cpu().numpy())
    if len(extra_points) > INIT_LABELS:
        break
if len(active_points) > 0:
    _, _, M = np.array(active_points).shape
    extra_points = np.expand_dims(np.array(extra_points), axis=1)
    extra_labels = np.array(extra_labels)
    active_points = np.concatenate([active_points, extra_points[:INIT_LABELS]])
    active_labels = np.concatenate([active_labels, extra_labels[:INIT_LABELS]])
else:
    active_points = extra_points[:INIT_LABELS]
    active_labels = extra_labels[:INIT_LABELS]

"""
# Add a lot of complementary points
extra_points = []
extra_labels = []
netE = networks['encoder']
from torch.autograd import Variable
for (images, labels, _) in train_dataloader:
    z = netE(Variable(images)).data.cpu().numpy()
    extra_points.extend(z)
    # Pick a label, any label... except the real one
    K = 10
    labels = (labels + np.random.randint(1, K)) % K
    extra_labels.extend(labels.cpu().numpy())
    if len(extra_points) > 3000:
        break
complementary_points = np.array(extra_points)
complementary_labels = np.array(extra_labels)
"""

training_len = 4000 + len(active_points) + len(complementary_points)
if len(active_points) < training_len:
    print("Padding active label dataset for training stability")
    active_points, active_labels = augment_to_length(active_points, active_labels, required_len=training_len)

print("Re-training classifier {} using {} active-learning label points".format(
    classifier_name, len(active_points) + len(complementary_points)))

MAX_EPOCHS = 25
for classifier_epoch in range(MAX_EPOCHS):
    # Apply learning rate decay and train for one pseudo-epoch
    for optimizer in optimizers.values():
        optimizer.param_groups[0]['lr'] = .0002 * (.9 ** classifier_epoch)
    start_time = time.time()
    train_active_learning(networks, optimizers, active_points, active_labels, complementary_points, complementary_labels, **options)
    print("Ran train_active_learning in {:.3f}s".format(time.time() - start_time))

# Evaluate against the test set
print("Evaluating {}".format(foldname))
start_time = time.time()
new_results = evaluate_classifier(networks, test_dataloader, verbose=False, fold=foldname, skip_reconstruction=True, **options)
print("Ran evaluate_classifier in {:.3f}s".format(time.time() - start_time))

print("Results:")
pprint(new_results)

print("Trained with {} active points, {} negative points".format(len(active_points), len(complementary_points)))
save_networks({classifier_name: networks[classifier_name]}, epoch=current_epoch, result_dir=options['result_dir'])
save_evaluation(new_results, options['result_dir'], get_current_epoch(options['result_dir']))
