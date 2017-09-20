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

print("Loaded Models:")
print(netE)
print(netG)
print(netD)


optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


dataloader = CustomDataloader(
        opt.dataset,
        batch_size=opt.batchSize,
        height=opt.imageSize,
        width=opt.imageSize,
        random_horizontal_flip=False,
        torch=True)


real_input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize).cuda()
noise = torch.FloatTensor(opt.batchSize, opt.latentSize).cuda()
fixed_noise = Variable(torch.FloatTensor(opt.batchSize, opt.latentSize).normal_(0, 1)).cuda()
label_one = torch.FloatTensor(opt.batchSize).cuda().fill_(1)
label_zero = torch.FloatTensor(opt.batchSize).cuda().fill_(0)
label_minus_one = torch.FloatTensor(opt.batchSize).cuda().fill_(-1)

for epoch in range(opt.epochs):
    for i, img_batch in enumerate(dataloader):
        ############################
        # (1) Update D network
        # WGAN: maximize D(G(z)) - D(x)
        ###########################
        for _ in range(5):
            netD.zero_grad()
            D_real_output = netD(Variable(img_batch))
            errD_real = D_real_output.mean()
            errD_real.backward(label_one)

            noise.normal_(0, 1)
            fake = netG(Variable(noise))
            D_fake_output = netD(fake.detach())
            errD_fake = D_fake_output.mean()
            errD_fake.backward(label_minus_one)

            gradient_penalty = calc_gradient_penalty(netD, img_batch, fake.data)
            gradient_penalty.backward()

            optimizerD.step()
        ###########################

        ############################
        # (2) Update G network:
        # WGAN: minimize D(G(z))
        ############################
        netG.zero_grad()
        DG_fake_output = netD(fake)
        errG = DG_fake_output.mean()
        errG.backward(label_one)
        optimizerG.step()
        ###########################

        ############################
        # (3) Update G(E()) network:
        # Autoencoder: Minimize X - G(E(X))
        ############################
        netE.zero_grad()
        netG.zero_grad()
        encoded = netE(Variable(img_batch))
        reconstructed = netG(encoded)
        errE = torch.mean(torch.abs(reconstructed - Variable(img_batch)))
        errE.backward()
        optimizerE.step()
        optimizerG.step()
        ############################

        D_x = D_real_output.data.mean()
        errD = errD_real + errD_fake
        if i % 25 == 0:
            print('[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} Loss_GP: {:.4f}  Loss_E: {:.4f}'.format(
                  epoch, opt.epochs, i, len(dataloader),
                  errD.data[0], errG.data[0], gradient_penalty.data[0], errE.data[0]))
            video_filename = "{}/generated.mjpeg".format(opt.resultDir)
            caption = "Epoch {}".format(epoch)
            demo_img = netG(fixed_noise)
            show(demo_img, video_filename=video_filename, caption=caption, display=False)
        if i % 100 == 0:
            show(img_batch, display=True, save=False)
            show(reconstructed, display=True, save=False)
            show(demo_img, display=True, save=False)

    # Apply learning rate decay
    optimizerE.param_groups[0]['lr'] *= .5
    optimizerD.param_groups[0]['lr'] *= .5
    optimizerG.param_groups[0]['lr'] *= .5

    # do checkpointing
    torch.save(netE.state_dict(), '%s/netE_epoch_%d.pth' % (opt.resultDir, epoch))
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.resultDir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.resultDir, epoch))
