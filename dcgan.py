from __future__ import print_function
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
import torchvision.utils as vutils
from torch.autograd import Variable
from imutil import show


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--ganType', default='dcgan', help='dcgan, wgan, wgan-gp')

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
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.conv1 = nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        self.conv4 = nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.conv5 = nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(True)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU(True)(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = nn.Tanh()(x)
        return x


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv5(x)

        if opt.ganType == 'dcgan':
            x = nn.Sigmoid()(x)
        elif opt.ganType == 'wgan':
            pass

        return x.view(-1, 1).squeeze(1)


netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

real_input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label_one = torch.FloatTensor(opt.batchSize)
label_zero = torch.FloatTensor(opt.batchSize)
label_minus_one = torch.FloatTensor(opt.batchSize)

if opt.cuda:
    # TODO: This is silly
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    real_input = real_input.cuda()
    label_one = label_one.cuda()
    label_zero = label_zero.cuda()
    label_minus_one = label_minus_one.cuda()
    noise = noise.cuda()
    fixed_noise = fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

if opt.ganType == 'dcgan':
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.ganType == 'wgan':
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr)

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        real_images, labels = data
        if opt.cuda:
            real_images = real_images.cuda()
        # Every batch can be a different size!
        batch_size = real_images.size(0)

        real_input.resize_as_(real_images).copy_(real_images)
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        label_one.resize_(batch_size)
        label_one.fill_(1)
        label_zero.resize_(batch_size)
        label_zero.fill_(0)
        label_minus_one.resize_(batch_size)
        label_minus_one.fill_(-1)

        if opt.ganType == 'wgan':
            for p in netD.parameters():
                p.data.clamp_(-.01, .01)

        ############################
        # (1) Update D network
        # DCGAN: maximize log(D(x)) + log(1 - D(G(z)))
        # WGAN: maximize D(G(z)) - D(x)
        ###########################
        critic_updates = 3 if opt.ganType == 'wgan' else 1
        for _ in range(critic_updates):
            netD.zero_grad()

            D_real_output = netD(Variable(real_input))
            if opt.ganType == 'dcgan':
                errD_real = criterion(D_real_output, Variable(label_one))
                errD_real.backward()
            elif opt.ganType == 'wgan':
                errD_real = D_real_output
                errD_real.backward(label_one)

            fake = netG(Variable(noise))
            D_fake_output = netD(fake.detach())
            if opt.ganType == 'dcgan':
                errD_fake = criterion(D_fake_output, Variable(label_zero))
                errD_fake.backward()
            elif opt.ganType == 'wgan':
                errD_fake = D_fake_output
                errD_fake.backward(label_minus_one)

            optimizerD.step()
        ###########################

        ############################
        # (2) Update G network:
        # DCGAN: maximize log(D(G(z)))
        # WGAN: minimize D(G(z))
        ############################
        netG.zero_grad()
        DG_fake_output = netD(fake)
        if opt.ganType == 'dcgan':
            errG = criterion(DG_fake_output, Variable(label_one))
            errG.backward()
        elif opt.ganType == 'wgan':
            errG = DG_fake_output
            errG.backward(label_one)
        optimizerG.step()
        ###########################

        D_x = D_real_output.data.mean()
        D_G_z1 = D_fake_output.data.mean()
        D_G_z2 = DG_fake_output.data.mean()
        errD = errD_real + errD_fake
        if i % 25 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            demo_img = netG(fixed_noise)
            show(demo_img, video_filename="generated.mjpeg", display=False)
        if i % 250 == 0:
            show(demo_img, display=True, save=False)

    # Apply learning rate decay
    optimizerD.param_groups[0]['lr'] *= .99
    optimizerG.param_groups[0]['lr'] *= .99

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
