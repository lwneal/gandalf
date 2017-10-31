#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys

def is_true(x):
    return not not x and x not in ['false', 'False', '0']

# Dataset (input) and result_dir (output) are always required
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

# Core Options: these determine the shape/size of the neural network
parser.add_argument('--dataset', help='Input filename (must be in .dataset format)')
parser.add_argument('--encoder', help='Name of encoder network')
parser.add_argument('--generator', help='Name of generator network')
parser.add_argument('--discriminator', help='Name of discriminator network')
parser.add_argument('--latent_size', type=int, default=100, help='Size of the latent z vector [default: 100]')
parser.add_argument('--image_size', type=int, default=64, help='Height / width of images [default: 64]')

# Other options are specific to training
parser.add_argument('--batch_size', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, [default: 0.0001]')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. [default: 0.5]')
parser.add_argument('--decay', type=float, default=0.9, help='Learning rate decay per epoch. [default: 0.9]')
parser.add_argument('--weight_decay', type=float, default=1.0, help='Optimizer L2 weight decay [default: 1.0]')
parser.add_argument('--random_horizontal_flip', type=is_true, default=False, help='Flip images during training. [default: False]')
parser.add_argument('--delete_background', type=is_true, default=False, help='Delete non-foreground pixels from images [default: False]')
parser.add_argument('--supervised_encoder', type=is_true, default=False, help='Train the encoder jointly with the classifier. [default: False]')
# Perceptual loss at 9 layers (VGG16 relu_2_2) following Johnson et al https://arxiv.org/abs/1603.08155
parser.add_argument('--perceptual_loss', type=is_true, default=False, help='Enable P-loss [default: False]')
parser.add_argument('--perceptual_depth', type=int, default=9, help='Number of layers of perceptual loss [default: 9]')
# Gradient penalty lambda defaults to 10 following Gulrajani et al https://arxiv.org/abs/1704.00028
parser.add_argument('--gradient_penalty_lambda', type=float, default=10.0, help='Magnitude of discriminator regularization [default: 10.0]')
parser.add_argument('--autoencoder_weight', type=float, default=1.0, help='Autoencoder training weight [default: 1.0]')
parser.add_argument('--gan_weight', type=float, default=1.0, help='GAN training weight [default: 1.0]')

# TODO: replace this, it's a hack
parser.add_argument('--attributes_only', type=is_true, default=False, help='Learn attributes, not classes [default: False]')

# This might change with each run
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for [default: 10]')

options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from training import train_counterfactual
from networks import build_networks, save_networks, get_optimizers
from options import save_options, load_options, get_current_epoch
from locking import acquire_lock, release_lock
from imutil import encode_video

if os.path.exists(options['result_dir']):
    options = load_options(options)

dataloader = CustomDataloader(fold='train', **options)
networks = build_networks(dataloader.num_classes, dataloader.num_attributes, **options)
optimizers = get_optimizers(networks, **options)

save_options(options)
start_epoch = get_current_epoch(options['result_dir']) + 1
acquire_lock(options['result_dir'])
try:
    for epoch in range(start_epoch, start_epoch + options['epochs']):
        # Apply learning rate decay
        for name, optimizer in optimizers.items():
            # Hack: Attribute networks need a higher learning rate
            if name == 'attribute':
                lr = options['lr'] * (options['decay'] ** epoch) * dataloader.num_attributes
            else:
                lr = options['lr'] * (options['decay'] ** epoch)
            optimizer.param_groups[0]['lr'] = lr

        # Train for one epoch
        video_filename = train_counterfactual(networks, optimizers, dataloader, epoch=epoch, **options)
        save_networks(networks, epoch, options['result_dir'])
finally:
    release_lock(options['result_dir'])
    encode_video(video_filename)
