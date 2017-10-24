#!/bin/bash
TARGET_DIR=mnist_28x28_e24de477
ORACLE_DIR=mnist_28x28_e2a0b34a

python experiments/generate_counterfactual.py --result_dir /mnt/results/$TARGET_DIR --mode uncertainty

python experiments/oracle.py --result_dir /mnt/results/$TARGET_DIR --oracle_result_dir /mnt/results/$ORACLE_DIR

python experiments/train_active_classifier.py --result_dir /mnt/results/$TARGET_DIR 
