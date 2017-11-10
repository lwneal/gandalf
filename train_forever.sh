#!/bin/bash
INIT_LABEL_COUNT=$1
while true; do 
    python experiments/train_active_classifier.py --result_dir /mnt/results/super_mnist --init_label_count $INIT_LABEL_COUNT --classifier_epochs 10
sleep 5; done
