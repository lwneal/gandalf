#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
import torch
from pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from training import train_adversarial_autoencoder
from networks import build_networks, get_optimizers
from options import save_options, load_options

# Dataset (input) and result_dir (output) are always required
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Input filename (must be in .dataset format)')
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

# Core Options: these determine the shape/size of the neural network.
core_options = ['encoder', 'generator', 'discriminator', 'latent_size', 'image_size']
parser.add_argument('--encoder', help='Name of encoder network')
parser.add_argument('--generator', help='Name of generator network')
parser.add_argument('--discriminator', help='Name of discriminator network')
parser.add_argument('--latent_size', type=int, help='Size of the latent z vector')
parser.add_argument('--image_size', type=int, help='Height / width of images')

# Other options can change with every run
parser.add_argument('--batch_size', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for [default: 25]')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, [default: 0.0001]')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. [default: 0.5]')
parser.add_argument('--decay', type=float, default=0.9, help='Learning rate decay per epoch. [default: 0.9]')

options = vars(parser.parse_args())

if os.path.exists(options['result_dir']):
    options = load_options(options, core_options)
else:
    save_options(options, core_options)

dataloader = CustomDataloader(random_horizontal_flip=False, **options)
networks = build_networks(dataloader.num_classes, **options)
optimizers = get_optimizers(networks, **options)

for epoch in range(options['epochs']):
    train_adversarial_autoencoder(networks, optimizers, dataloader, epoch=epoch, **options)

    # Apply learning rate decay
    for optimizer in optimizers.values():
        optimizer.param_groups[0]['lr'] *= options['decay']

    # do checkpointing
    for name in networks:
        net = networks[name]
        torch.save(net.state_dict(), '{}/{}_epoch_{:02d}.pth'.format(options['result_dir'], name, epoch))
