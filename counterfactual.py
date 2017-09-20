from __future__ import print_function
import network_definitions
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import grad
from torch.autograd import Variable

from imutil import show
from dataloader import CustomDataloader
from gradient_penalty import calc_gradient_penalty

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Path to a .dataset file')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--latentSize', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--netE', default='', help="path to netE (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--resultDir', default='.', help='Output directory for images and model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.resultDir)
except OSError:
    pass

cudnn.benchmark = True


netE = network_definitions.encoderLReLU64(opt.latentSize)
if opt.netE:
    netE.load_state_dict(torch.load(opt.netE))

netG = network_definitions.generatorReLU64(opt.latentSize)
if opt.netG:
    netG.load_state_dict(torch.load(opt.netG))

netD = network_definitions.discriminatorLReLU64()
if opt.netD:
    netD.load_state_dict(torch.load(opt.netD))

networks = {
    'encoder': netE,
    'generator': netG,
    'discriminator': netD,
}

def get_optimizers(networks, lr, beta1):
    optimizers = {}
    for name in networks:
        net = networks[name]
        optimizers[name] = optim.Adam(net.parameters(), lr=lr, betas=(beta1, .999))
    return optimizers

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
    from training import train_adversarial_autoencoder
    train_adversarial_autoencoder(networks, optimizers, dataloader, **params)

    # Apply learning rate decay
    for opt in optimizers:
        opt.param_groups[0]['lr'] *= .9

    # do checkpointing
    torch.save(netE.state_dict(), '%s/netE_epoch_%d.pth' % (opt.resultDir, epoch))
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.resultDir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.resultDir, epoch))
