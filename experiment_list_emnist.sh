#!/bin/bash

RESULT_DIR=emnist_28x28_a0c89197
ORACLE_DIR=emnist_28x28_6ee4135d

./run_experiment_1.sh $RESULT_DIR  $ORACLE_DIR   random  False
cp /mnt/results/$RESULT_DIR/eval_epoch_0025.json emnist_semisupervised.json

./run_experiment_1.sh $RESULT_DIR  $ORACLE_DIR   uncertainty  False    
cp /mnt/results/$RESULT_DIR/eval_epoch_0025.json emnist_uncertainty_sampling.json

./run_experiment_1.sh $RESULT_DIR  $ORACLE_DIR   random  True    
cp /mnt/results/$RESULT_DIR/eval_epoch_0025.json emnist_counterfactual_random.json

./run_experiment_1.sh $RESULT_DIR  $ORACLE_DIR   uncertainty  True    
cp /mnt/results/$RESULT_DIR/eval_epoch_0025.json emnist_counterfactual_uncertainty.json
