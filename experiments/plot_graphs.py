#!/usr/bin/env python
from __future__ import print_function
import json
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks
from options import load_options, get_current_epoch
from evaluation import evaluate_classifier

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

options = vars(parser.parse_args())

result_dir = options['result_dir']

filenames = os.listdir(result_dir) 
filenames = [f for f in filenames if f.startswith('eval') and f.endswith('.json')]
filenames = sorted(filenames)
filenames = [os.path.join(result_dir, f) for f in filenames]


data = {
        "test_accuracy": [],
        "test_mae": [],
        "test_mse": [],
        "train_accuracy": [],
        "train_mae": [],
        "train_mse": [],
}

for f in filenames:
    d = json.load(open(f))
    print(f)
    print(d)
    data['train_accuracy'].append(d['train']['accuracy'])
    data['train_mae'].append(d['train']['mae'])
    data['train_mse'].append(d['train']['mse'])
    data['test_accuracy'].append(d['test']['accuracy'])
    data['test_mae'].append(d['test']['mae'])
    data['test_mse'].append(d['test']['mse'])

import pandas as pd
df = pd.DataFrame(data)

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt


from imutil import show
for title in data:
    plot = df.plot(y=title)
    plot.set_title('{} per epoch'.format(title))
    show(plot)
