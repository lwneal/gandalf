#!/bin/bash

RESULT_DIR=emnist_28x28_a0c89197
ORACLE_DIR=emnist_28x28_6ee4135d

./run_experiment_1.sh $RESULT_DIR  $ORACLE_DIR   random  False
mv /mnt/results/$RESULT_DIR/trajectories /mnt/results/$RESULT_DIR/trajectories_baseline_semisupervised
cp /mnt/results/$RESULT_DIR/eval_epoch_0025.json emnist_semisupervised.json

./run_experiment_1.sh $RESULT_DIR  $ORACLE_DIR   uncertainty  False    
mv /mnt/results/$RESULT_DIR/trajectories /mnt/results/$RESULT_DIR/trajectories_baseline_uncertainty_sampling
cp /mnt/results/$RESULT_DIR/eval_epoch_0025.json emnist_uncertainty_sampling.json

./run_experiment_1.sh $RESULT_DIR  $ORACLE_DIR   random  True    
mv /mnt/results/$RESULT_DIR/trajectories /mnt/results/$RESULT_DIR/trajectories_counterfactual_random
cp /mnt/results/$RESULT_DIR/eval_epoch_0025.json emnist_counterfactual_random.json

./run_experiment_1.sh $RESULT_DIR  $ORACLE_DIR   uncertainty  True    
mv /mnt/results/$RESULT_DIR/trajectories /mnt/results/$RESULT_DIR/trajectories_counterfactual_uncertainty
cp /mnt/results/$RESULT_DIR/eval_epoch_0025.json emnist_counterfactual_uncertainty.json
