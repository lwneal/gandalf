#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
from pprint import pprint
import sys

# Dataset (input) and result_dir (output) are always required
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

# Core Options: these determine the shape/size of the neural network
parser.add_argument('--dataset', help='Input filename (must be in .dataset format)')
parser.add_argument('--evaluation_epoch', type=int, help='Epoch to load for training')
parser.add_argument('--example_count', type=int, default=100, help='Number of labels to train with [default: 100]')
parser.add_argument('--classifier_epochs', type=int, default=50, help='Max number of training epochs [default: 50]')

options = vars(parser.parse_args())


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from training import train_classifier
from evaluation import evaluate_classifier, save_evaluation
from networks import build_networks, get_optimizers
from options import save_options, load_options, get_current_epoch
from locking import acquire_lock, release_lock


loaded_options = load_options(options)
evaluation_epoch = options['evaluation_epoch']
if not evaluation_epoch:
    evaluation_epoch = get_current_epoch(options['result_dir'])
# A limited-size training dataset
print("Loading a training dataset with only {} random examples".format(options['example_count']))
train_dataloader = CustomDataloader(fold='train', **options)

# A larger test dataset for evaluation
# TODO: This should be the validation fold, or a hold-out set from fold='train'
print("Loading a full-sized test dataset...")
test_dataloader_options = options.copy()
del test_dataloader_options['example_count']
test_dataloader = CustomDataloader(fold='test', **test_dataloader_options)

networks = build_networks(train_dataloader.num_classes, epoch=evaluation_epoch, load_classifier=False, **options)
optimizers = get_optimizers(networks, **options)

print("Training classifier using {} labels".format(train_dataloader.dsf.count()))
for classifier_epoch in range(options['classifier_epochs']):
    # Apply learning rate decay
    for optimizer in optimizers.values():
        optimizer.param_groups[0]['lr'] = options['lr'] * (options['decay'] ** classifier_epoch)
    # Train for one epoch
    train_classifier(networks, optimizers, train_dataloader, epoch=classifier_epoch, **options)

foldname = '{}_example_count_{:06d}'.format(test_dataloader.dsf.name, options['example_count'])
classifier_options = options.copy()
classifier_options['fold'] = foldname

print("Evaluating classifier and saving evaluation as {}".format(foldname))
new_results = evaluate_classifier(networks, test_dataloader, verbose=False, **classifier_options)

print("Parsing relevant statistics from evaluation run...")
interesting_statistics = ['accuracy']
for fold in new_results:
    for key in list(new_results[fold]):
        if key not in interesting_statistics:
            del new_results[fold][key]

print("Saving evaluation statistics:")
pprint(new_results)
save_evaluation(new_results, options['result_dir'], evaluation_epoch)
