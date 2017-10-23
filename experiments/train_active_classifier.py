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

print("TODO: Load every labeled trajectory from trajectories/ and labels/")

print("# TODO: Use those labeled points to re-train the classifier")

print("# TODO: Save the re-trained active_learning_classifier.pth")
classifier_name = options['classifier_name']
save_networks({classifier_name: networks[classifier_name]}, epoch=current_epoch, result_dir=options['result_dir'])

