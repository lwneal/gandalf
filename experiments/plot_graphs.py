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
parser.add_argument('--fold', default='test', help='One of: train, test')

options = vars(parser.parse_args())
options = load_options(options)

result_dir = options['result_dir']

filenames = os.listdir(result_dir) 
filenames = [f for f in filenames if f.startswith('eval') and f.endswith('.json')]
filenames = sorted(filenames)
filenames = [os.path.join(result_dir, f) for f in filenames]


fold = options['fold']

data = {
        fold + "_accuracy": [],
        fold + "_mae": [],
        fold + "_mse": [],
}

for f in filenames:
    d = json.load(open(f))
    print(f)
    print(d)
    data[fold + '_accuracy'].append(d[fold]['accuracy'])
    data[fold + '_mae'].append(d[fold]['mae'])
    data[fold +'_mse'].append(d[fold]['mse'])

import pandas as pd
df = pd.DataFrame(data)

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt


from imutil import show
for statistic in data:
    plot = df.plot(y=statistic)
    dataset_name = options['dataset'].split('/')[-1].replace(".dataset", '')
    statistic_label = statistic.replace('_', ' ').title()
    name = 'Dataset: {}  Latent Size: {}'.format(dataset_name, options['latent_size'])
    plot.set_title('{}\n{} per epoch'.format(name, statistic_label))
    plot.set_ylabel(statistic_label)
    plot.set_xlabel("Epoch")
    filename = "plot_{}".format(statistic)
    filename = os.path.join(result_dir, filename)
    plot.figure.savefig(filename)
