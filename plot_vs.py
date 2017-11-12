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
    title = "MNIST/EMNIST 10-class, Human Oracle"
    plotting.compare_multiple(sys.argv[1:], names, 'output-figure-emnist.png', title=title, prefix='comparison_emnist-digits_uncertainty_sampling')

    title = "MNIST 10-class, Human Oracle"
    plotting.compare_multiple(sys.argv[1:], names, 'output-figure-mnist.png', title=title)
