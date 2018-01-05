import time
import os
import torch
import torch.nn as nn
import random
import numpy as np
from torchvision import models
from torch.autograd import Variable
from gradient_penalty import calc_gradient_penalty
from torch.nn.functional import nll_loss, binary_cross_entropy
import torch.nn.functional as F
from torch.nn.functional import softmax, log_softmax, relu
import imutil
from dataloader import FlexibleCustomDataloader

np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)


def log_sum_exp(inputs, dim=None, keepdim=False):
    return inputs - log_softmax(inputs, dim=1)


def train_model(networks, optimizers, dataloader, epoch=None, **options):
    for net in networks.values():
        net.train()
    netD = networks['discriminator']
    netG = networks['generator']
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']
    discriminator_per_gen = options['discriminator_per_gen']

    noise = Variable(torch.FloatTensor(batch_size, latent_size).cuda())
    fixed_noise = Variable(torch.FloatTensor(batch_size, latent_size).normal_(0, 1)).cuda()
    clamp_to_unit_sphere(fixed_noise)
    demo_images, demo_labels = next(d for d in dataloader)

    dataset_filename = os.path.join(options['result_dir'], 'aux_dataset.dataset')
    use_aux_dataset = os.path.exists(dataset_filename) # and options['use_aux_dataset']
    aux_kwargs = {
        'dataset': dataset_filename,
        'batch_size': options['batch_size'],
        'image_size': options['image_size'],
    }
    if use_aux_dataset:
        print("Enabling aux dataset")
        aux_dataloader = FlexibleCustomDataloader(**aux_kwargs)

    start_time = time.time()
    correct = 0
    total = 0

    for i, (images, class_labels) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(class_labels)

        ############################
        # Generator Updates
        ############################
        if i % discriminator_per_gen == 0:
            netG.zero_grad()
            gen_images = netG(gen_noise(noise))
            
            # Feature Matching: Average of one batch of real vs. generated
            features_real = netD(images, return_features=True).mean(dim=0)
            features_gen = netD(gen_images, return_features=True).mean(dim=0)

            errG = torch.mean((features_real - features_gen) ** 2)
            errG.backward()
            optimizerG.step()
        ###########################

        ############################
        # Discriminator Updates
        ###########################
        netD.zero_grad()

        # Classify generated examples as the K+1th "open" class
        fake_images = netG(gen_noise(noise)).detach()
        fake_logits = netD(fake_images)
        err_fake = F.softplus(log_sum_exp(fake_logits)).mean()
        errD = err_fake
        errD.backward()

        # Classify real examples as their classes
        real_logits = netD(images)
        positive_labels = (labels == 1).type(torch.cuda.FloatTensor)
        errHingePos = (F.softplus(-real_logits) * positive_labels).mean()

        errC = errHingePos
        errC.backward()

        optimizerD.step()
        ############################

        # Keep track of accuracy on positive-labeled examples for monitoring
        _, pred_idx = real_logits.max(1)
        _, label_idx = labels.max(1)
        correct += sum(pred_idx == label_idx).data.cpu().numpy()[0]
        total += len(labels)

        if i % 100 == 0:
            demo_fakes = netG(fixed_noise)
            img = torch.cat([demo_fakes.data[:36]])
            filename = "{}/demo_{}.jpg".format(result_dir, int(time.time()))
            imutil.show(img, filename=filename, resize_to=(512,512))

            bps = (i+1) / (time.time() - start_time)
            ed = errD.data[0]
            eg = errG.data[0]
            ec = errC.data[0]
            acc = correct / max(total, 1)
            msg = '[{}][{}/{}] D:{:.3f} G:{:.3f} C:{:.3f} Acc. {:.3f} {:.3f} batch/sec'
            msg = msg.format(
                  epoch, i+1, len(dataloader),
                  ed, eg, ec, acc, bps)
            print(msg)
            #print("errHingePos: {:.3f}".format(errHingePos.data[0]))
            print("err_fake {:.3f}".format(err_fake.data[0]))
            #print("err_real {:.3f}".format(err_real.data[0]))
            print("Accuracy {}/{}".format(correct, total))
    return True


def gen_noise(noise, spherical_noise=True):
    noise.data.normal_(0, 1)
    if spherical_noise:
        noise = clamp_to_unit_sphere(noise)
    return noise


def clamp_to_unit_sphere(x):
    norm = torch.norm(x, p=2, dim=1)
    norm = norm.expand(1, x.size()[0])
    return torch.mul(x, 1/norm.t())
