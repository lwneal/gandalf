#!/bin/bash

python average_eval_files.py /mnt/results/mnist_uncertainty_sampling/eval_uncertainty_monday*.json > uncertainty_sampling-monday.json

python average_eval_files.py /mnt/results/mnist_counterfactual_top1/eval_random_monday_*.json > random-monday.json
python average_eval_files.py /mnt/results/mnist_counterfactual_top1/eval_random-nearest_monday_*.json > random_nearest-monday.json
python average_eval_files.py /mnt/results/mnist_counterfactual_top1/eval_uncertainty-nearest_monday_*.json > uncertainty_nearest-monday.json
python average_eval_files.py /mnt/results/mnist_counterfactual_top1/eval_uncertainty-random_monday_*.json > uncertainty_random-monday.json

python average_eval_files.py /mnt/results/mnist_counterfactual_certainty/eval_certainty_random_monday*.json > certainty_random-monday.json
python average_eval_files.py /mnt/results/mnist_counterfactual_certainty/eval_certainty_furthest_monday*.json > certainty_furthest-monday.json

python plot_graph.py
