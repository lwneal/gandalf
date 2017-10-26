import sys
import json
import plotting
from imutil import show


def plot_active_learning(filename):
    try:
        evaluations = json.load(open(filename))
    except:
        print("Error: could not load JSON from file {}".format(filename))
        raise
    keys = sorted([k for k in evaluations if k.startswith('active_trajectories')])
    x = [int(k.split('_')[-1]) for k in keys]
    y = [evaluations[k]['accuracy'] for k in keys]
    plot = plotting.plot_xy(x, y, x_axis="Number of Examples", y_axis="Accuracy", title=filename)
    plot_filename = "{}.jpg".format(filename.replace('.json', ''))
    show(plot, filename=plot_filename)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: {} eval_epoch_0123.json".format(sys.argv[0]))
        exit()
    plot_active_learning(sys.argv[1])
