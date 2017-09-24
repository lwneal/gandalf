#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from training import train_adversarial_autoencoder
from networks import build_networks, save_networks, get_optimizers
from options import save_options, load_options, get_current_epoch

# Dataset (input) and result_dir (output) are always required
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Input filename (must be in .dataset format)')
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

# Core Options: these determine the shape/size of the neural network.
core_options = ['encoder', 'generator', 'discriminator', 'latent_size', 'image_size']
parser.add_argument('--encoder', help='Name of encoder network')
parser.add_argument('--generator', help='Name of generator network')
parser.add_argument('--discriminator', help='Name of discriminator network')
parser.add_argument('--latent_size', type=int, default=100, help='Size of the latent z vector [default: 100]')
parser.add_argument('--image_size', type=int, default=64, help='Height / width of images [default: 64]')

# Other options can change with every run
parser.add_argument('--batch_size', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for [default: 25]')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, [default: 0.0001]')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. [default: 0.5]')
parser.add_argument('--decay', type=float, default=0.9, help='Learning rate decay per epoch. [default: 0.9]')
parser.add_argument('--random_horizontal_flip', type=bool, default=False, help='Flip images during training. [default: False]')
parser.add_argument('--delete_background', type=bool, default=False, help='Delete non-foreground pixels from images [default: False]')
parser.add_argument('--joint_encoder_classifier', type=bool, default=False, help='Train encoder/classifier jointly. [default: False]')

options = vars(parser.parse_args())

if os.path.exists(options['result_dir']):
    options = load_options(options, core_options)

dataloader = CustomDataloader(fold='train', **options)
networks = build_networks(dataloader.num_classes, **options)
optimizers = get_optimizers(networks, **options)

save_options(options, core_options)

start_epoch = get_current_epoch(options['result_dir']) + 1

for epoch in range(start_epoch, start_epoch + options['epochs']):
    # Apply learning rate decay
    for optimizer in optimizers.values():
        optimizer.param_groups[0]['lr'] = options['lr'] * (options['decay'] ** epoch)
    # Train for one epoch
    train_adversarial_autoencoder(networks, optimizers, dataloader, epoch=epoch, **options)
    save_networks(networks, epoch, options['result_dir'])
