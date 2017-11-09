import sys
import json
import plotting
from imutil import show
import plotting


if __name__ == '__main__':
    #names = [n.split('/')[-1] for n in sys.argv[1:]]
    names = sys.argv[1:]
    #title = "Open Set MNIST vs EMNIST: 1000 initial points"
    #plotting.compare_multiple(sys.argv[1:], names, 'output.png', title=title, prefix='openset')
    title = "MNIST Accuracy: Human-Labeled Counterfactuals"
    plotting.compare_multiple(sys.argv[1:], names, 'mnist-human-counterfactual.png', title=title)
