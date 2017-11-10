#!/bin/bash
while true; do 
    python experiments/evaluate_openset.py --result_dir /mnt/results/mnist_28x28_cycleception/ --classifier_name active_learning_classifier --comparison_dataset /mnt/data/emnist.dataset
    sleep 10
done
