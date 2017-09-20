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


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Path to a .dataset file')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--latentSize', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size()[0], 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    ones = torch.ones(disc_interpolates.size()).cuda()
    gradients = grad(
            outputs=disc_interpolates, 
            inputs=interpolates, 
            grad_outputs=ones, 
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True)[0]

    LAMBDA = 20.
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return penalty



netG = network_definitions.deconvReLU64(opt.latentSize)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = network_definitions.convLReLU64()
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

real_input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize).cuda()
noise = torch.FloatTensor(opt.batchSize, opt.latentSize, 1, 1).cuda()
fixed_noise = Variable(torch.FloatTensor(opt.batchSize, opt.latentSize, 1, 1).normal_(0, 1)).cuda()
label_one = torch.FloatTensor(opt.batchSize).cuda().fill_(1)
label_zero = torch.FloatTensor(opt.batchSize).cuda().fill_(0)
label_minus_one = torch.FloatTensor(opt.batchSize).cuda().fill_(-1)

netD.cuda()
netG.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

from dataloader import CustomDataloader
dataloader = CustomDataloader(
        opt.dataset,
        batch_size=opt.batchSize,
        height=opt.imageSize,
        width=opt.imageSize,
        random_horizontal_flip=True,
        torch=True)

for epoch in range(opt.epochs):
    for i, img_batch in enumerate(dataloader):
        ############################
        # (1) Update D network
        # DCGAN: maximize log(D(x)) + log(1 - D(G(z)))
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
        # DCGAN: maximize log(D(G(z)))
        # WGAN: minimize D(G(z))
        ############################
        netG.zero_grad()
        DG_fake_output = netD(fake)
        errG = DG_fake_output.mean()
        errG.backward(label_one)
        optimizerG.step()
        ###########################

        D_x = D_real_output.data.mean()
        D_G_z1 = D_fake_output.data.mean()
        D_G_z2 = DG_fake_output.data.mean()
        errD = errD_real + errD_fake
        if i % 5 == 0:
            print('[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}'.format(
                  epoch, opt.epochs, i, len(dataloader), errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            video_filename = "{}/generated.mjpeg".format(opt.outf)
            caption = "Epoch {}".format(epoch)
            demo_img = netG(fixed_noise)
            show(demo_img, video_filename=video_filename, caption=caption, display=False)
        if i % 25 == 0:
            show(demo_img, display=True, save=False)

    # Apply learning rate decay
    optimizerD.param_groups[0]['lr'] *= .5
    optimizerG.param_groups[0]['lr'] *= .5

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
