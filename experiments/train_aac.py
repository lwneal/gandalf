#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
import torch
import json
from pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from training import train_adversarial_autoencoder
from networks import build_networks, get_optimizers


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
    # We are resuming an old experiment: load core options from params.json
    print("Resuming existing experiment at {}".format(options['result_dir']))
    old_opts = json.load(open(os.path.join(options['result_dir'], 'params.json')))
    pprint(old_opts)
    for opt_name in core_options:
        options[opt_name] = old_opts[opt_name]
        print("Setting {} to {}".format(opt_name, options[opt_name]))
else:
    # We are starting a new experiment: save core options to params.json
    print("Beginning a new experiment at {}".format(options['result_dir']))
    for opt_name in core_options:
        if not options.get(opt_name):
            print("Error: required option --{} was not specified".format(opt_name))
            exit()
    print("Creating directory {}".format(options['result_dir']))
    os.makedirs(options['result_dir'])
    with open(os.path.join(options['result_dir'], 'params.json'), 'w') as fp:
        json.dump(options, fp)

dataloader = CustomDataloader(
        options['dataset'],
        batch_size=options['batch_size'],
        height=options['image_size'],
        width=options['image_size'],
        random_horizontal_flip=False,
        torch=True)

num_classes = dataloader.num_classes
print("Building classifier with {} classes".format(num_classes))
networks = build_networks(
        options['latent_size'],
        options['result_dir'],
        options['image_size'],
        num_classes,
        options['encoder'],
        options['generator'],
        options['discriminator'])
optimizers = get_optimizers(networks, options['lr'], options['beta1'])


for epoch in range(options['epochs']):
    params = {
        'latent_size': options['latent_size'],
        'batch_size': options['batch_size'],
        'result_dir': options['result_dir'],
        'epochs': options['epochs'],
        'image_size': options['image_size'],
    }
    train_adversarial_autoencoder(networks, optimizers, dataloader, epoch=epoch, **params)

    # Apply learning rate decay
    for optimizer in optimizers.values():
        optimizer.param_groups[0]['lr'] *= .9

    # do checkpointing
    for name in networks:
        net = networks[name]
        torch.save(net.state_dict(), '{}/{}_epoch_{:02d}.pth'.format(options['result_dir'], name, epoch))
