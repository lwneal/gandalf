#!/bin/bash

./run_experiment_1.sh SVHN_32x32_2a57d7ce  SVHN_32x32_8f0bd2c3   random  False
mv /mnt/results/SVHN_32x32_2a57d7ce/trajectories /mnt/results/SVHN_32x32_2a57d7ce/trajectories_random
cp /mnt/results/SVHN_32x32_2a57d7ce/eval_epoch_0025.json svhn_semisupervised.json

./run_experiment_1.sh SVHN_32x32_2a57d7ce  SVHN_32x32_8f0bd2c3   uncertainty  False    
mv /mnt/results/SVHN_32x32_2a57d7ce/trajectories /mnt/results/SVHN_32x32_2a57d7ce/trajectories_uncertainty_sampling
cp /mnt/results/SVHN_32x32_2a57d7ce/eval_epoch_0025.json svhn_uncertainty_sampling.json

./run_experiment_1.sh SVHN_32x32_2a57d7ce  SVHN_32x32_8f0bd2c3   random  True    
mv /mnt/results/SVHN_32x32_2a57d7ce/trajectories /mnt/results/SVHN_32x32_2a57d7ce/trajectories_counterfactual_random
cp /mnt/results/SVHN_32x32_2a57d7ce/eval_epoch_0025.json svhn_counterfactual_random.json

./run_experiment_1.sh SVHN_32x32_2a57d7ce  SVHN_32x32_8f0bd2c3   uncertainty  True    
mv /mnt/results/SVHN_32x32_2a57d7ce/trajectories /mnt/results/SVHN_32x32_2a57d7ce/trajectories_counterfactual_uncertainty
cp /mnt/results/SVHN_32x32_2a57d7ce/eval_epoch_0025.json svhn_counterfactual_uncertainty.json
