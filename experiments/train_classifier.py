#!/usr/bin/env python
import time
import random
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
parser.add_argument('--experiment_type', type=str, default='baseline', help='One of: semisupervised, uncertainty_sampling, counterfactual')
parser.add_argument('--classifier_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--best_epoch', type=is_true, default=False, help='Select best-fit epoch (only use for oracle training)')
parser.add_argument('--save', type=is_true, default=True, help='If set to False, do not save network')

parser.add_argument('--weight_decay', type=float, default=1.0, help='Weight decay [default: 1.0]')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks, get_optimizers, save_networks
from options import load_options, get_current_epoch
from evaluation import evaluate_classifier, save_evaluation
from training import train_classifier

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


def load_labels(result_dir, label_count=None):
    active_label_dir = os.path.join(result_dir, 'labels')
    if not os.path.exists(active_label_dir):
        print("No labels available in {}".format(active_label_dir))
        return []
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


def load_trajectory(result_dir, trajectory_id):
    trajectory_dir = os.path.join(result_dir, 'trajectories')
    if not os.path.exists(trajectory_dir):
        os.mkdir(trajectory_dir)
    trajectory_filenames = os.listdir(trajectory_dir)
    for filename in trajectory_filenames:
        if filename.endswith('.npy') and trajectory_id in filename:
            return np.load(os.path.join(trajectory_dir, filename))
    print("Warning: Failed to find trajectory {}.npy".format(trajectory_id))
    return None


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


import numpy as np
# Labels are always a JSON list of dicts
result_dir = options['result_dir']
labels = load_labels(result_dir)

# In the Grid labeling experiment the "trajectories" are actually just images
images = []
image_labels = []
for label_info in labels:
    image_batch = load_trajectory(result_dir, label_info['trajectory_id'])
    images.extend(image_batch)

    # TODO: this is hard-coded for 10-class MNIST, should generalize
    assert len(image_batch) == 100
    pos_neg = [int(c) for c in label_info['labels'].split(',')]
    assert len(pos_neg) == 100
    class_indices = list(range(10)) * 10
    label_data = np.zeros((10*10,10))
    for i, idx in enumerate(class_indices):
        val = 1 if pos_neg[i] > 0 else -1
        label_data[i,idx] = val
    image_labels.extend(label_data)

images = np.array(images)
image_labels = np.array(image_labels).astype(int)

foldname = 'grid_{:08d}'.format(len(images))

best_epoch = 0
best_acc = 0
for classifier_epoch in range(options['classifier_epochs']):
    # Apply learning rate decay and train for one pseudo-epoch
    for optimizer in optimizers.values():
        optimizer.param_groups[0]['lr'] = .01 * (.95 ** classifier_epoch)
    start_time = time.time()
    train_classifier(networks, optimizers, images, image_labels, **options)
    print("Ran train_classifier in {:.3f}s".format(time.time() - start_time))

    if options['best_epoch'] == False and classifier_epoch < options['classifier_epochs'] - 1:
        continue

    # Evaluate against the test set
    print("Evaluating {}".format(foldname))
    start_time = time.time()
    new_results = evaluate_classifier(networks, test_dataloader, verbose=False, fold=foldname, skip_reconstruction=True, **options)
    print("Ran evaluate_classifier in {:.3f}s".format(time.time() - start_time))

    print("Results:")
    pprint(new_results)
    if new_results[foldname]['accuracy'] > best_acc:
        best_acc = new_results[foldname]['accuracy']
        best_results = new_results
        best_epoch = classifier_epoch
        if options['save']:
            print("Saving network with accuracy {}".format(best_acc))
            save_networks({classifier_name: networks[classifier_name]}, epoch=current_epoch, result_dir=options['result_dir'])

if options['save']:
    save_evaluation(best_results, options['result_dir'], get_current_epoch(options['result_dir']))
print("Best Results:")
pprint(best_results)
print("Best epoch: {}".format(best_epoch))
