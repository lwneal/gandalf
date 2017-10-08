import time
import torch
import torch.nn as nn
import random
import numpy as np
from torchvision import models
from torch.autograd import Variable
from gradient_penalty import calc_gradient_penalty
from torch.nn.functional import nll_loss
import imutil


def train_adversarial_autoencoder(networks, optimizers, dataloader, epoch=None, **options):
    netD = networks['discriminator']
    netG = networks['generator']
    netE = networks['encoder']
    netC = networks['classifier']
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    optimizerE = optimizers['encoder']
    optimizerC = optimizers['classifier']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    if options['perceptual_loss']:
        vgg16 = models.vgg.vgg16(pretrained=True)
        P_layers = list(vgg16.features.children())
        P_layers = P_layers[:options['perceptual_depth']]
        netP = nn.Sequential(*P_layers)
        netP.cuda()

    noise = torch.FloatTensor(batch_size, latent_size).cuda()
    fixed_noise = Variable(torch.FloatTensor(batch_size, latent_size).normal_(0, 1)).cuda()
    label_one = torch.FloatTensor(batch_size).cuda().fill_(1)
    label_zero = torch.FloatTensor(batch_size).cuda().fill_(0)
    label_minus_one = torch.FloatTensor(batch_size).cuda().fill_(-1)
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(labels)
        ############################
        # (1) Update D network
        # WGAN: maximize D(G(z)) - D(x)
        ###########################
        for _ in range(5):
            netD.zero_grad()
            D_real_output = netD(images)
            errD_real = D_real_output.mean()
            errD_real.backward(label_one)

            noise.normal_(0, 1)
            fake = netG(Variable(noise))
            fake = fake.detach()
            D_fake_output = netD(fake)
            errD_fake = D_fake_output.mean()
            errD_fake.backward(label_minus_one)

            gradient_penalty = calc_gradient_penalty(netD, images.data, fake.data, options['gradient_penalty_lambda'])
            gradient_penalty.backward()

            optimizerD.step()
        ###########################

        # Zero gradient for all networks
        netG.zero_grad()
        netE.zero_grad()
        netC.zero_grad()

        ############################
        # (2) Update G network:
        # WGAN: minimize D(G(z))
        ############################
        noise.normal_(0, 1)
        fake = netG(Variable(noise))
        DG_fake_output = netD(fake)
        errG = DG_fake_output.mean()
        errG.backward(label_one)
        ###########################

        ############################
        # (3) Update G(E()) network:
        # Autoencoder: Minimize X - G(E(X))
        ############################
        encoded = netE(images)
        reconstructed = netG(encoded)
        weight = options['autoencoder_lambda']
        errGE = weight * torch.mean(torch.abs(reconstructed - images))
        if options['perceptual_loss']:
            errGE += weight * torch.mean(torch.abs(netP(reconstructed) - netP(images)))
        errGE.backward()
        ############################

        ############################
        # (4) Update E(G()) network:
        # Inverse Autoencoder: Minimize Z - E(G(Z))
        ############################
        noise.normal_(0, 1)
        fake = netG(Variable(noise))
        reencoded = netE(fake)
        errEG = torch.mean((reencoded - Variable(noise)) ** 2)
        errEG.backward()
        ############################

        if options['joint_encoder_classifier']:
            optimizerE.step()

        ############################
        # (5) Update C(Z) network:
        # Categorical Cross-Entropy
        ############################
        latent_points = netE(images)
        class_predictions = netC(latent_points)
        errC = nll_loss(class_predictions, labels)
        errC.backward()
        ############################

        if not options['joint_encoder_classifier']:
            optimizerE.step()
        optimizerG.step()
        optimizerC.step()

        # https://discuss.pytorch.org/t/argmax-with-pytorch/1528/2
        _, predicted = class_predictions.max(1)
        correct += sum(predicted.data == labels.data)
        total += len(predicted)

        errD = errD_real + errD_fake
        if i % 25 == 0:
            msg = '[{}][{}/{}] D:{:.3f} G:{:.3f} GP:{:.3f} GE:{:.3f} EG:{:.3f} EC: {:.3f} C_acc:{:.3f}'
            msg = msg.format(
                  epoch, i, len(dataloader),
                  errD.data[0],
                  errG.data[0],
                  gradient_penalty.data[0],
                  errGE.data[0],
                  errEG.data[0],
                  errC.data[0],
                  float(correct) / total)
            print(msg)
            video_filename = "{}/generated.mjpeg".format(result_dir)
            caption = "Epoch {:04d} iter {:05d}".format(epoch, i)
            demo_gen = netG(fixed_noise)
            imutil.show(demo_gen, video_filename=video_filename, caption=caption, display=False)
        if i % 100 == 0:
            img = torch.cat([images[:12], reconstructed.data[:12], demo_gen.data[:12]])
            filename = "{}/demo_{}.jpg".format(result_dir, int(time.time()))
            imutil.show(img, caption=msg, font_size=8, filename=filename)


def train_classifier(networks, optimizers, dataloader, **options):
    netD = networks['discriminator']
    netG = networks['generator']
    netE = networks['encoder']
    netC = networks['classifier']
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    optimizerE = optimizers['encoder']
    optimizerC = optimizers['classifier']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(labels)

        ############################
        # Update C(Z) network:
        # Categorical Cross-Entropy
        ############################
        latent_points = netE(images)
        class_predictions = netC(latent_points)
        errC = nll_loss(class_predictions, labels)
        errC.backward()
        optimizerC.step()
        ############################

        _, predicted = class_predictions.max(1)
        correct += sum(predicted.data == labels.data)
        total += len(predicted)

        if i % 25 == 0 or i == len(dataloader) - 1:
            print('[{}/{}] Classifier Loss: {:.3f} Classifier Accuracy:{:.3f}'.format(
                i, len(dataloader), errC.data[0], float(correct) / total))
    return float(correct) / total


def shuffle(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def train_active_learning(networks, optimizers, active_points, active_labels, **options):
    netC = networks['classifier']
    optimizerC = optimizers['classifier']
    result_dir = options['result_dir']
    latent_size = options['latent_size']
    batch_size = options['batch_size']

    correct = 0
    total = 0

    def generator():
        assert len(active_points) == len(active_labels)
        i = 0
        shuffle(active_points, active_labels)
        while i < len(active_points):
            x = torch.FloatTensor(active_points[i:i+batch_size])
            y = torch.LongTensor(active_labels[i:i+batch_size])
            yield x.squeeze(1), y
            i += batch_size
    dataloader = generator()

    
    for i, (latent_points, labels) in enumerate(dataloader):
        latent_points = Variable(latent_points)
        labels = Variable(labels)

        latent_points = latent_points.cuda()
        labels = labels.cuda()

        ############################
        # Update C(Z) network:
        # Categorical Cross-Entropy
        ############################
        class_predictions = netC(latent_points)
        errC = nll_loss(class_predictions, labels)
        errC.backward()
        optimizerC.step()
        ############################

        _, predicted = class_predictions.max(1)
        correct += sum(predicted.data == labels.data)
        total += len(predicted)

    print('[{}/{}] Classifier Loss: {:.3f} Classifier Accuracy:{:.3f}'.format(
        i, len(active_points) / batch_size, errC.data[0], float(correct) / total))
    return float(correct) / total
