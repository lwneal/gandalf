import json
import numpy as np
import pandas as pd
from imutil import show

# Hack to keep matplotlib from crashing when run without X
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Apply sane defaults to matplotlib
import seaborn as sns
sns.set_style('darkgrid')


class NoDataAvailable(Exception):
    pass


def plot_xy(x, y, x_axis="X", y_axis="Y", title="Plot"):
    df = pd.DataFrame({'x': x, 'y': y})
    plot = df.plot(x='x', y='y')

    plot.grid(b=True, which='major')
    plot.grid(b=True, which='minor')
    
    plot.set_title(title)
    plot.set_ylabel(y_axis)
    plot.set_xlabel(x_axis)
    return plot


def parse_active_learning_series(eval_filename):
    try:
        evaluations = json.load(open(eval_filename))
    except:
        print("Error: could not load JSON from file {}".format(eval_filename))
        raise NoDataAvailable
    keys = sorted([k for k in evaluations if k.startswith('active_trajectories')])
    if len(keys) == 0:
        raise NoDataAvailable
    x = [int(k.split('_')[-1]) for k in keys]
    y = [evaluations[k]['accuracy'] for k in keys]
    return x, y


def plot_active_learning(eval_filename="results_epoch_0025.json"):
    try:
        x, y = parse_active_learning_series(eval_filename)
    except NoDataAvailable:
        return None
    plot = plot_xy(x, y, x_axis="Number of Examples", y_axis="Accuracy", title=eval_filename)
    plot_filename = "{}.jpg".format(eval_filename.replace('.json', ''))
    show(plot, filename=plot_filename)
    return plot


def compare_active_learning(eval_filename, baseline_eval_filename):
    try:
        x, y = parse_active_learning_series(eval_filename)
        x2, y2 = parse_active_learning_series(baseline_eval_filename)
    except NoDataAvailable:
        return None
    
    plt.plot(x, y, "g") # this method
    plt.plot(x2, y2, "b") # baseline
    this_approach_name = eval_filename.split('/')
    plt.suptitle('Accuracy Vs Baseline')
    plt.ylabel('Accuracy (10 classes)')
    plt.xlabel('Number of Queries')
    plt.legend(['This Method', 'Baseline'])

    fig_filename = eval_filename.replace('.json', '-vs-baseline.png')
    plt.savefig(fig_filename)
    show(fig_filename)
    return fig_filename
