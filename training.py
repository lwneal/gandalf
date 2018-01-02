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
from torch.nn.functional import softmax, log_softmax, relu
import imutil
from dataloader import FlexibleCustomDataloader

np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

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
    discriminator_updates_per_generator = options['discriminator_per_gen']
    video_filename = "{}/generated.mjpeg".format(result_dir)

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
        if i % discriminator_updates_per_generator == 0:
            netG.zero_grad()
            noise = gen_noise(noise)
            errG = -netD(netG(noise)).max(dim=1)[0].mean()
            errG *= options['gan_weight']
            errG.backward()
            optimizerG.step()
        ###########################


        ############################
        # Discriminator Updates
        ###########################
        netD.zero_grad()

        # Train classifier hinge loss
        errC = train_discriminator(images, labels, netD, netG, noise)

        # Every generated image is a negative example
        noise = gen_noise(noise)
        fake_images = netG(noise).detach()
        errD = relu(1 + netD(fake_images)).sum()

        # Apply WGAN-GP gradient penalty
        errGP = calc_gradient_penalty(netD, images.data, fake_images.data)

        if use_aux_dataset:
            aux_images, aux_labels = aux_dataloader.get_batch()
            aux_images = Variable(aux_images)
            aux_labels = Variable(aux_labels)
            errC += train_discriminator(aux_images, aux_labels, netD, netG, noise)

        total_error = errC + errD + errGP
        total_error.backward()
        optimizerD.step()
        ############################

        # Keep track of accuracy on positive-labeled examples for monitoring
        net_y = netD(images)
        _, pred_idx = net_y.max(1)
        _, label_idx = labels.max(1)
        correct += sum(pred_idx == label_idx).data.cpu().numpy()[0]
        total += len(labels)


        if i % 100 == 0:
            demo_fakes = netG(fixed_noise)
            img = torch.cat([demo_fakes.data[:36]])
            filename = "{}/demo_{}.jpg".format(result_dir, int(time.time()))
            imutil.show(img, filename=filename, resize_to=(512,512))

            msg = '[{}][{}/{}] GrP {:.3f} D:{:.3f} G:{:.3f} Acc. {:.3f} {:.3f} batch/sec'
            bps = i / (time.time() - start_time)
            msg = msg.format(
                  epoch, i, len(dataloader),
                  errGP.data[0],
                  errD.data[0],
                  errG.data[0],
                  correct / max(total, 1),
                  bps)
            print(msg)
    return video_filename


def train_discriminator(images, labels, netD, netG, noise):
    # Train discriminator as a classifier
    net_y = netD(images)
    positive_labels = (labels == 1).type(torch.cuda.FloatTensor)
    negative_labels = (labels == -1).type(torch.cuda.FloatTensor)

    # Hinge loss term for negative and positive labels
    #errHinge = relu(1+net_y) * negative_labels + relu(1-net_y) * positive_labels
    errHinge = relu(1+net_y) * negative_labels + relu(1-net_y) * positive_labels
    # Log-Likelihood to calibrate the K separate one-vs-all classifiers
    errNLL = -log_softmax(net_y, dim=1) * positive_labels
    errC = errHinge.sum() + errNLL.sum()

    return errC


def show_weights(net):
    from imutil import show
    for layer in net.children():
        if hasattr(layer, 'weight'):
            print("\nLayer {}".format(layer))
            show(layer.weight.data, save=False)
            print('Weight sum: {}'.format(to_scalar(layer.weight.abs().sum())))
            print('Weight min: {}'.format(to_scalar(layer.weight.min())))
            print('Weight max: {}'.format(to_scalar(layer.weight.max())))


def abs_sum(variable):
    weight_sum = variable.abs().sum()
    return to_scalar(weight_sun)

def to_scalar(variable):
    return variable.data.cpu().numpy()[0]

def gen_noise(noise, spherical_noise=True):
    noise.data.normal_(0, 1)
    if spherical_noise:
        noise = clamp_to_unit_sphere(noise)
    return noise


def clamp_to_unit_sphere(x):
    norm = torch.norm(x, p=2, dim=1)
    norm = norm.expand(1, x.size()[0])
    return torch.mul(x, 1/norm.t())


def shuffle(*args):
    rng_state = np.random.get_state()
    for arg in args:
        np.random.shuffle(arg)
        np.random.set_state(rng_state)


def train_classifier(networks, optimizers, images, labels, **options):
    netC = networks[options['classifier_name']]
    netE = networks['encoder']
    netG = networks['generator']
    for net in networks.values():
        net.train()
    # Do not update the generator
    netG.eval()

    optimizerC = optimizers[options['classifier_name']]
    optimizerE = optimizers['encoder']
    result_dir = options['result_dir']
    latent_size = options['latent_size']
    batch_size = options['batch_size']
    num_classes = 10  # TODO

    def generator(points, labels):
        assert len(points) == len(labels)
        while True:
            i = 0
            shuffle(points, labels)
            while i < len(points) - batch_size:
                x = torch.FloatTensor(points[i:i+batch_size].transpose((0,3,1,2)))
                y = torch.FloatTensor(labels[i:i+batch_size].astype(float))
                x, y = x.cuda(), y.cuda()
                yield x.squeeze(1), y
                i += batch_size

    dataloader = generator(images, labels)

    correct = 0
    total = 0
    batch_count = len(images) // batch_size
    for i in range(batch_count):
        images, labels = next(dataloader)
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        maxval, _ = labels.max(dim=1)
        is_blue = (maxval == 1).type(torch.cuda.FloatTensor)
        is_red = (1 - is_blue)

        net_y = netC(netE(images))[:,:-1]

        label_pos = (labels == 1).type(torch.cuda.FloatTensor)
        label_neg = (labels == -1).type(torch.cuda.FloatTensor)

        # Positive and Negative labels: train a 1-vs-all classifier for each class
        from torch.nn.functional import log_softmax
        log_preds = log_softmax(net_y)
        errClass = -(log_preds * (labels == 1).type(torch.cuda.FloatTensor)).sum()

        # Additional term: Hinge loss on the linear layer before the activation
        from torch.nn.functional import relu
        errHinge = (relu(1+net_y) * label_neg + relu(1-net_y) * label_pos).sum()

        errC = errClass + errHinge

        errC.backward()
        optimizerC.step()

        class_preds = torch.exp(log_preds)
        _, predicted = class_preds.max(1)
        _, correct_labels = labels.max(1)
        correct += sum((predicted.data == correct_labels.data).type(torch.cuda.FloatTensor) * is_blue.data)
        total += sum(is_blue.data)

    print('[{}/{}] LogLoss: {:.3f}  HingeLoss: {:.3f} Accuracy:{:.3f}'.format(
        i, batch_count, errClass.data[0], errHinge.data[0],float(correct) / total))

    return float(correct) / total
