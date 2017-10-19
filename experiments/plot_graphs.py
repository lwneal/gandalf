#!/usr/bin/env python
from __future__ import print_function
import json
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

options = vars(parser.parse_args())


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks
from options import load_options, get_current_epoch
from evaluation import evaluate_classifier
import pandas as pd

# Hack to apply sane defaults to matplotlib
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt


options = load_options(options)

result_dir = options['result_dir']

filenames = os.listdir(result_dir) 
filenames = [f for f in filenames if f.startswith('eval') and f.endswith('.json')]
filenames = sorted(filenames)
filenames = [os.path.join(result_dir, f) for f in filenames]

evals = []
for f in filenames:
    evals.append(json.load(open(f)))

print("Loaded {} evaluation reports".format(len(evals)))
folds = []
statistics = []
data = {}
for e in evals:
    for fold in e.keys():
        folds.append(fold)
        for statistic in e[fold]:
            statistics.append(statistic)
            key = '{}_{}'.format(fold, statistic)
            if key not in data:
                data[key] = []
            data[key].append(e[fold][statistic])

unique_folds = set(folds)
unique_statistics = set(statistics)
print("Got data for {} folds and {} statistics".format(len(unique_folds), len(unique_statistics)))


def plot(sequence, statistic):
    if len(sequence) < 2:
        print("Skipping short sequence {}".format(statistic))
        return
    print("Plotting {}".format(statistic))
    df = pd.DataFrame({statistic: sequence})
    plot = df.plot(y=statistic)
    dataset_name = options['dataset'].split('/')[-1].replace(".dataset", '')
    statistic_label = statistic.replace('_', ' ').title()
    name = 'Dataset: {}  Latent Size: {}'.format(dataset_name, options['latent_size'])
    plot.set_title('{}\n{} per epoch'.format(name, statistic_label))
    plot.set_ylabel(statistic_label)
    plot.set_xlabel("Epoch")
    filename = "plot_{}.png".format(statistic)
    filename = os.path.join(result_dir, filename)
    plot.figure.savefig(filename)


max_len = max([len(v) for v in data.values()])
for k in data:
    plot(data[k], k)

