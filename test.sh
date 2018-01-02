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

rm -rf /mnt/results/svhn_32x32_test

#python datasets/download_svhn.py

experiments/train_model.py \
    --result_dir /mnt/results/svhn_32x32_test \
    --dataset /mnt/data/svhn.dataset \
    --image_size 32 \
    --latent_size 10 \
    --epochs 1

experiments/example.py --result_dir /mnt/results/svhn_32x32_test

#python experiments/test_dataloader.py
