#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks
from options import load_options
from evaluation import evaluate_classifier

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')
parser.add_argument('--fold', default="test", help='Name of evaluation fold [default: test]')

options = vars(parser.parse_args())

options = load_options(options)

dataloader = CustomDataloader(last_batch=True, **options)
networks = build_networks(dataloader.num_classes, **options)

evaluate_classifier(networks, dataloader, **options)
