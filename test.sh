#!/bin/bash
set -e

# TODO: Make sure everything is installed correctly

function make_sure_directory_exists() {
    if [ ! -d $1 ]; then
        echo "Directory $1 does not exist, creating it now"
        mkdir -p $1
    else
        echo "Directory $1 exists"
    fi
}

make_sure_directory_exists /mnt/results
make_sure_directory_exists /mnt/data

python datasets/download_mnist.py

experiments/train_aac.py \
    --result_dir /mnt/results/mnist_28x28 \
    --dataset /mnt/data/mnist.dataset \
    --image_size 28 \
    --latent_size 10 \
    --encoder encoder28 \
    --generator generator28 \
    --discriminator discriminator28 \
    --epochs 1

experiments/example.py --result_dir /mnt/results/mnist_28x28

python experiments/test_dataloader.py
