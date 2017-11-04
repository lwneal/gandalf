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
parser.add_argument('--positive_margin', type=int, default=1, help='Number of positive points near the boundary to discard')
parser.add_argument('--negative_margin', type=int, default=2, help='Number of negative points near the boundary to discard')

parser.add_argument('--use_complementary_labels', type=is_true, default=True, help='If False, ignore all complementary labels')
parser.add_argument('--weight_decay', type=float, default=1.0, help='Weight decay [default: 1.0]')
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


def load_active_learning_trajectories(labels, train_dataloader, positive_margin, negative_margin):
    active_points = []
    active_labels = []
    complementary_points = []
    complementary_labels = []
    for label in labels:
        trajectory_filename = get_trajectory_filename(result_dir, label['trajectory_id'])
        points = np.load(trajectory_filename)

        # Two-Point Decision Boundary Labeling
        #  On the left, positive labels for the left class
        #  On the right, positive labels for the right class
        #  In the middle, negative (complementary) labels for both classes
        left_class = train_dataloader.lab_conv.idx[label['start_class']]
        right_class = train_dataloader.lab_conv.idx[label['target_class']]
        left_boundary = int(label['start_label_point']) if label['start_label_point'] else 0
        right_boundary = int(label['end_label_point'])


        left_points = slice_with_margin(points, 0, left_boundary, right_margin=positive_margin)
        center_points = slice_with_margin(points, left_boundary, right_boundary, negative_margin, negative_margin)
        right_points = slice_with_margin(points, right_boundary, len(points), left_margin=positive_margin)

        active_points.extend(left_points)
        active_labels.extend([left_class] * len(left_points))
        active_points.extend(right_points)
        active_labels.extend([right_class] * len(right_points))

        # For each center point we have two complementary labels: not left_class, not right_class
        complementary_points.extend(center_points)
        complementary_labels.extend([left_class] * len(center_points))
        complementary_points.extend(center_points)
        complementary_labels.extend([right_class] * len(center_points))

    active_points = np.expand_dims(np.array(active_points), axis=1)
    active_labels = np.array(active_labels)
    complementary_points = np.expand_dims(np.array(complementary_points), axis=1)
    complementary_labels = np.array(complementary_labels)
    return active_points, active_labels, complementary_points, complementary_labels


def slice_with_margin(array, left, right, left_margin=0, right_margin=0):
    left_idx = min(left + left_margin, len(array))
    right_idx = max(0, right - right_margin)
    return np.squeeze(array[left_idx:right_idx], axis=1)


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
    active_points, active_labels, complementary_points, complementary_labels = load_active_learning_trajectories(labels, train_dataloader, positive_margin=options['positive_margin'], negative_margin=options['negative_margin'])
elif options['experiment_type'] == 'uncertainty_sampling':
    active_points, active_labels = load_active_learning_points(labels, train_dataloader)
    complementary_points = np.array([])
    complementary_labels = np.array([])
elif options['experiment_type'] == 'semisupervised':
    active_points, active_labels = load_active_learning_points(labels, train_dataloader)
    complementary_points = np.array([])
    complementary_labels = np.array([])
elif options['experiment_type'] == 'baseline':
    active_points = np.array([])
    active_labels = np.array([])
    complementary_points = np.array([])
    complementary_labels = np.array([])
else:
    raise ValueError("Unknown experiment_type")
print("Loaded {} trajectories totalling {} positive points and {} negative points".format(
	len(labels), len(active_points), len(complementary_points)))

# Every run starts with a set of eg. 1k initial labels
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

training_len = 30 * 1000
if len(active_points) < training_len:
    print("Padding active label dataset for training stability")
    active_points, active_labels = augment_to_length(active_points, active_labels, required_len=training_len)
active_points = np.array(active_points)
active_labels = np.array(active_labels)

def to_torch(x):
    from torch import FloatTensor
    from torch.autograd import Variable
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    return Variable(FloatTensor(x)).cuda()

def gen(z):
    from imutil import show
    netG = networks['generator']
    show(netG(to_torch(z)))


################################################
# Every class is a convex hull in latent space
################################################

# Sort the positive labeled points
label_indices = active_labels.argsort()
active_labels = active_labels[label_indices]
active_points = active_points[label_indices]

rightmost = [0]
for i in range(len(active_labels) - 1):
    if active_labels[i] != active_labels[i+1]:
        rightmost.append(i)
rightmost.append(len(active_labels) - 1)

interpolated_points = []
interpolated_labels = []
for i in range(10000):
    cls = random.randint(0, len(rightmost) - 2)
    left_idx, right_idx = (rightmost[cls], rightmost[cls+1])
    a = active_points[np.random.randint(left_idx, right_idx)]
    b = active_points[np.random.randint(left_idx, right_idx)]
    theta = np.random.random()
    zerp = theta * a + (1 - theta) * b
    #zerp /= np.linalg.norm(zerp)
    interpolated_points.append(zerp)
    interpolated_labels.append(cls)
active_labels = np.concatenate([active_labels, interpolated_labels])
active_points = np.concatenate([active_points, interpolated_points])

################################################



print("Re-training classifier {} using {} active-learning label points".format(
    classifier_name, len(active_points) + len(complementary_points)))

best_epoch = 0
best_acc = 0
for classifier_epoch in range(options['classifier_epochs']):
    # Apply learning rate decay and train for one pseudo-epoch
    for optimizer in optimizers.values():
        optimizer.param_groups[0]['lr'] = .0001 * (.9 ** classifier_epoch)
    start_time = time.time()
    train_active_learning(networks, optimizers, active_points, active_labels, complementary_points, complementary_labels, **options)
    print("Ran train_active_learning in {:.3f}s".format(time.time() - start_time))

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
        print("Saving network with accuracy {}".format(best_acc))
        save_networks({classifier_name: networks[classifier_name]}, epoch=current_epoch, result_dir=options['result_dir'])

print("Trained with {} active points, {} negative points".format(len(active_points), len(complementary_points)))
if options['save']:
    save_evaluation(best_results, options['result_dir'], get_current_epoch(options['result_dir']))
print("Best Results:")
pprint(best_results)
print("Best epoch: {}".format(best_epoch))
