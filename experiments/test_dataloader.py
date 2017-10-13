#!/usr/bin/env python
import time
import argparse
import os
import sys
from pprint import pprint

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
parser.add_argument('--foo', help='Bar')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader


def time_dataset(dataset_name):
    print("Timing how long it takes to load dataset {}".format(dataset_name))
    start_time = time.time()
    dataloader = CustomDataloader(dataset=dataset_name, **options)
    batch_count = 0
    for example in dataloader:
        batch_count += 1
    time_taken = time.time() - start_time
    print("Loaded {} examples from {} in {:.3f}s".format(
        dataloader.count(), dataset_name, time_taken))
    print("Image size: {0}x{0} \tTime per image: {1:.3f}s".format(
        dataloader.lab_conv.image_size, time_taken / dataloader.count()))
    return time_taken


dataset_downloaders = os.listdir('datasets')
datasets = [name.replace('download_', '').replace('.py', '') for name in dataset_downloaders]
print("There are {} available datasets:".format(len(datasets)))
pprint(datasets)
for dataset_name in datasets:
    time_dataset('/mnt/data/{}.dataset'.format(dataset_name))
