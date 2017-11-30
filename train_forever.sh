#!/bin/bash
while true; do
    python experiments/train_classifier.py --result_dir /mnt/results/super_mnist --classifier_epochs 100
    sleep 3
done
