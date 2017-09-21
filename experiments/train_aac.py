#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataloader import CustomDataloader
from training import train_adversarial_autoencoder
from networks import build_networks, get_optimizers


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Path to a .dataset file')
parser.add_argument('--batchSize', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--imageSize', type=int, default=64, help='Height / width of images [default: 64]')
parser.add_argument('--latentSize', type=int, default=100, help='Size of the latent z vector [default: 100]')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for [default: 25]')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, [default: 0.0001]')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. [default: 0.5]')
parser.add_argument('--decay', type=float, default=0.9, help='Learning rate decay per epoch. [default: 0.9]')
parser.add_argument('--netE', default='', help="path to netE (to continue training) [default: None]")
parser.add_argument('--netG', default='', help="path to netG (to continue training) [default: None]")
parser.add_argument('--netD', default='', help="path to netD (to continue training) [default: None]")
parser.add_argument('--resultDir', default='.', help='Output directory for images and model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.resultDir)
except OSError:
    pass


networks = build_networks(opt.latentSize)
optimizers = get_optimizers(networks, opt.lr, opt.beta1)

dataloader = CustomDataloader(
        opt.dataset,
        batch_size=opt.batchSize,
        height=opt.imageSize,
        width=opt.imageSize,
        random_horizontal_flip=False,
        torch=True)

for epoch in range(opt.epochs):
    params = {
        'latentSize': opt.latentSize,
        'batchSize': opt.batchSize,
        'resultDir': opt.resultDir,
        'epochs': opt.epochs,
        'imageSize': opt.imageSize,
    }
    train_adversarial_autoencoder(networks, optimizers, dataloader, epoch=epoch, **params)

    # Apply learning rate decay
    for optimizer in optimizers.values():
        optimizer.param_groups[0]['lr'] *= .9

    # do checkpointing
    for name in networks:
        net = networks[name]
        torch.save(net.state_dict(), '{}/{}_epoch_{:02d}.pth'.format(opt.resultDir, name, epoch))
