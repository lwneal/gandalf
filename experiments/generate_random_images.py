#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys

def is_true(x):
    return not not x and x.lower().startswith('t')

# Dataset (input) and result_dir (output) are always required
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

# Other options can change with every run
parser.add_argument('--batch_size', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--fold', type=str, default='validation', help='Fold [default: validation]')
parser.add_argument('--classifier_name', type=str, default='active_learning_classifier',
        help='Name of the classifier to use [default: active_learning_classifier]')
parser.add_argument('--start_epoch', type=int, help='Epoch to start from (defaults to most recent epoch)')
parser.add_argument('--count', type=int, default=1, help='Number of counterfactuals to generate')

options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks
from options import load_options
import numpy as np
import torch
from torch import autograd
from torch.autograd import Variable

def to_torch(z, requires_grad=False):
    return Variable(torch.FloatTensor(z), requires_grad=requires_grad).cuda()

def to_np(z):
    return z.data.cpu().numpy()

options = load_options(options)

# Active Learning trajectories can only be single examples
# TODO: enable batches
options['batch_size'] = 1

# NOTE: This dataloader should sample from a held-out pool of unlabeled examples used only for active learning
# In these experiments we use the 'validation' fold for this purpose
dataloader = CustomDataloader(**options)
networks = build_networks(dataloader.num_classes, **options)

netG = networks['generator']
netE = networks['encoder']
netC = networks['classifier']
netD = networks['discriminator']
result_dir = options['result_dir']
image_size = options['image_size']
latent_size = options['latent_size']
result_dir = options['result_dir']

#for i in range(options['count']):
	# For MNIST, SVHN
# Generate a 10x10 square visualization
N = 2#dataloader.num_classes

imagess = []
for _ in range(N*N):
    rand_vec = np.random.normal(size=(1,latent_size))
    mag = np.linalg.norm(rand_vec)
    z = rand_vec/mag
    img = netG(to_torch(z))
    imagess.append(img)

import pdb; pdb.set_trace()
images = np.array(imagess).transpose((0,2,3,1))

trajectory_id = '{}_{}.jpeg'.format(dataloader.dsf.name, int(time.time() * 1000))
path = os.path.join(result_dir, 'randomly_generated')
filename = os.path.join(path, trajectory_id)

if not os.path.exists(path):
    print("Creating randomly_generated directory {}".format(path))
    os.mkdir(path)
# Save the images in npy format to re-load as training data
np.save(filename, images)

# Save the images in jpg format to display to the user
imutil.show(images, filename=filename)