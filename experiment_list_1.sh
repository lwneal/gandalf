#!/bin/bash

# Totally stupid baseline: semi-supervised learning, random labels
./run_experiment_1.sh mnist_28x28_e24de477 mnist_28x28_e2a0b34a random False
cp /mnt/results/mnist_28x28_e24de477/eval_epoch_0025.json experiment_1_semisupervised.json

# Baseline: Active Uncertainty Sampling
./run_experiment_1.sh mnist_28x28_e24de477 mnist_28x28_e2a0b34a uncertainty False
cp /mnt/results/mnist_28x28_e24de477/eval_epoch_0025.json experiment_1_uncertainty_sampling.json

# Ablation: Counterfactual Trajectories w/ Random sampling
./run_experiment_1.sh mnist_28x28_e24de477 mnist_28x28_e2a0b34a random True
cp /mnt/results/mnist_28x28_e24de477/eval_epoch_0025.json experiment_1_counterfactual_random.json

# Full Model: Counterfactual Trajectories w/ Uncertainty Sampling
./run_experiment_1.sh mnist_28x28_e24de477 mnist_28x28_e2a0b34a uncertainty True
cp /mnt/results/mnist_28x28_e24de477/eval_epoch_0025.json experiment_1_counterfactual_uncertainty.json
