import time
import torch
import numpy as np
from torch import autograd
from torch.autograd import Variable
from torch.nn.functional import nll_loss, cross_entropy
import imutil


def generate_trajectory(networks, dataloader, **options):
    netG = networks['generator']
    netE = networks['encoder']
    netC = networks['classifier']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    real_input = torch.FloatTensor(batch_size, 3, image_size, image_size).cuda()
    noise = torch.FloatTensor(batch_size, latent_size).cuda()

    real_images, labels = dataloader.get_batch()
    real_images = Variable(real_images)
    
    # We start with vectors in the latent space Z
    z = to_np(netE(real_images))
    z = Variable(torch.FloatTensor(z), requires_grad=True).cuda()

    # We want to move them so their classification changes
    target_labels = torch.LongTensor(batch_size)
    desired_class = 0
    target_labels[:] = desired_class
    target_labels = Variable(target_labels).cuda()

    momentum = Variable(torch.zeros(z.size())).cuda()

    for i in range(1000):
        for _ in range(10):
            cf_loss = nll_loss(netC(z), target_labels)
            dc_dz = autograd.grad(cf_loss, z, cf_loss, retain_graph=True)[0]
            momentum -= dc_dz * 10.0
            z += momentum
            momentum *= .995
        print("Loss: {}".format(cf_loss.data[0]))
        print("Latent point: {}...".format(z[0].data.cpu().numpy()[:5]))
        print("Gradient: {}...".format(dc_dz[0].data.cpu().numpy()[:5]))
        print("Momentum: {}...".format(momentum[0].data.cpu().numpy()[:5]))
        classes = to_np(netC(z).max(1)[1])[:4]
        print("Class: {}...".format(classes))
        if all(classes == desired_class):
            break
        if i % 10 == 0:
            imutil.show(netG(z)[:4])
    return to_np(z)


def to_torch(z, requires_grad=False):
    return Variable(torch.FloatTensor(z), requires_grad=requires_grad).cuda()


def to_np(z):
    return z.data.cpu().numpy()
