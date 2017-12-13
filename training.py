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
from torch.nn.functional import softmax, log_softmax
import imutil
from dataloader import CustomDataloader

np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

def train_counterfactual(networks, optimizers, dataloader, epoch=None, **options):
    for net in networks.values():
        net.train()
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
    video_filename = "{}/generated.mjpeg".format(result_dir)

    # By convention, if it ends with 'sphere' it uses the unit sphere
    spherical = type(netE).__name__.endswith('sphere')

    noise = Variable(torch.FloatTensor(batch_size, latent_size).cuda())
    fixed_noise = Variable(torch.FloatTensor(batch_size, latent_size).normal_(0, 1)).cuda()
    if spherical:
        clamp_to_unit_sphere(fixed_noise)
    demo_images, demo_labels = next(d for d in dataloader)

    correct = 0
    total = 0

    dataset_filename = os.path.join(options['result_dir'], 'aux_dataset.dataset')
    use_aux_dataset = os.path.exists(dataset_filename) # and options['use_aux_dataset']
    aux_kwargs = {
        'dataset': dataset_filename,
        'batch_size': options['batch_size'],
        'image_size': options['image_size'],
    }
    if use_aux_dataset:
        print("Enabling aux dataset")
        aux_dataloader = CustomDataloader(**aux_kwargs)

    for i, (images, class_labels) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(class_labels)
        ############################
        # (1) Update D network
        ###########################
        # WGAN: maximize D(G(z)) - D(x)
        for _ in range(5):
            netD.zero_grad()
            noise = gen_noise(noise, spherical)
            fake_images = netG(noise).detach()
            errD = netD(images).mean() - netD(fake_images).mean()
            errD *= options['gan_weight']
            errD.backward()
            optimizerD.step()
        # Also update D network based on user-provided extra labels
        if use_aux_dataset:
            netD.zero_grad()
            aux_images, aux_labels = aux_dataloader.get_batch()
            aux_images = Variable(aux_images)
            aux_labels = Variable(aux_labels.type(torch.cuda.FloatTensor))
            alpha = len(aux_dataloader) / (len(dataloader) + len(aux_dataloader))
            d_aux = netD(aux_images)
            total_real = aux_labels.sum() + 1
            total_fake = (1 - aux_labels).sum() + 1
            errAuxD = ((d_aux * aux_labels).sum() / total_real 
                    - (d_aux * (1 - aux_labels)).sum() / total_fake)
            errAuxD.backward()
            optimizerD.step()

        # Apply WGAN-GP gradient penalty
        netD.zero_grad()
        errGP = calc_gradient_penalty(netD, images.data, fake_images.data)
        errGP.backward()
        optimizerD.step()
        ###########################

        ############################
        # Realism: minimize D(G(z))
        ############################
        netE.zero_grad()
        netG.zero_grad()
        noise = gen_noise(noise, spherical)
        errG = netD(netG(noise)).mean() * options['gan_weight']
        errG.backward()
        ###########################

        ############################
        # Consistency: minimize x - G(E(x))
        ############################
        noise = gen_noise(noise, spherical)
        hallucination = netG(noise)
        regenerated = netG(netE(hallucination))
        errEG = torch.mean(torch.abs(regenerated - hallucination))
        errEG *= options['autoencoder_weight']
        errEG.backward()
        optimizerG.step()
        optimizerE.step()
        ############################

        ############################
        # Classification: Max likelihood C(E(x))
        ############################
        netE.zero_grad()
        netC.zero_grad()
        preds = log_softmax(netC(netE(images)))
        errC = nll_loss(preds, labels)
        errC.backward()
        optimizerC.step()
        optimizerE.step()

        _, pred_idx = preds.max(1)
        correct += sum(pred_idx == labels).data.cpu().numpy()[0]
        total += len(labels)
        ############################

        if i % 100 == 0:
            print("Classifier Network Weights:")
            show_weights(netC)

            reconstructed = netG(netE(Variable(demo_images)))
            demo_fakes = netG(fixed_noise)
            img = torch.cat([demo_images[:12], reconstructed.data[:12], demo_fakes.data[:12]])
            filename = "{}/demo_{}.jpg".format(result_dir, int(time.time()))
            imutil.show(img, filename=filename)

            msg = '[{}][{}/{}] D:{:.3f} G:{:.3f} EG:{:.3f} EC: {:.3f} Acc. {:.3f}'
            msg = msg.format(
                  epoch, i, len(dataloader),
                  errD.data[0],
                  errG.data[0],
                  errEG.data[0],
                  errC.data[0],
                  correct / max(total, 1))
            print(msg)
    return video_filename


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

def gen_noise(noise, spherical_noise):
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
