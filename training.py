import time
import torch
import torch.nn as nn
import random
import numpy as np
from torchvision import models
from torch.autograd import Variable
from gradient_penalty import calc_gradient_penalty
from torch.nn.functional import nll_loss, binary_cross_entropy
import imutil


def train_counterfactual(networks, optimizers, dataloader, epoch=None, **options):
    for net in networks.values():
        net.train()
    netD = networks['discriminator']
    netG = networks['generator']
    netE = networks['encoder']
    netC = networks['classifier']
    netA = networks.get('attribute')
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    optimizerE = optimizers['encoder']
    optimizerC = optimizers['classifier']
    optimizerA = optimizers.get('attribute')
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']
    video_filename = "{}/generated.mjpeg".format(result_dir)

    if options['perceptual_loss']:
        vgg16 = models.vgg.vgg16(pretrained=True)
        P_layers = list(vgg16.features.children())
        P_layers = P_layers[:options['perceptual_depth']]
        netP = nn.Sequential(*P_layers)
        netP.cuda()

    # By convention, if it ends with 'sphere' it uses the unit sphere
    spherical = type(netE).__name__.endswith('sphere')

    noise = Variable(torch.FloatTensor(batch_size, latent_size).cuda())
    fixed_noise = Variable(torch.FloatTensor(batch_size, latent_size).normal_(0, 1)).cuda()
    if spherical:
        clamp_to_unit_sphere(fixed_noise)

    label_one = torch.FloatTensor(batch_size).cuda().fill_(1)
    label_zero = torch.FloatTensor(batch_size).cuda().fill_(0)
    label_minus_one = torch.FloatTensor(batch_size).cuda().fill_(-1)
    correct = 0
    total = 0
    attr_correct = 0
    attr_total = 0
    
    for i, (images, class_labels, attributes) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(class_labels)
        if netA:
            attributes = Variable(attributes)
        ############################
        # (1) Update D network
        # WGAN: maximize D(G(z)) - D(x)
        ###########################
        for _ in range(5):
            netD.zero_grad()
            noise = gen_noise(noise, spherical)
            fake_images = netG(noise).detach()
            errD = netD(images).mean() - netD(fake_images).mean()
            errD.backward()
            optimizerD.step()
        ###########################

        # WGAN-GP
        errGP = calc_gradient_penalty(netD, images.data, fake_images.data)
        errGP.backward()
        optimizerD.step()

        # Zero gradient for all networks
        netG.zero_grad()
        netE.zero_grad()

        ############################
        # Realism: minimize D(G(z))
        ############################
        noise = gen_noise(noise, spherical)
        errG = netD(netG(noise)).mean() * options['gan_weight']
        errG.backward()
        ###########################

        ############################
        # Consistency: minimize z - E(G(z))
        ############################
        noise = gen_noise(noise, spherical)
        fake = netG(noise)
        reencoded = netE(fake)
        errEG = torch.mean((reencoded - noise) ** 2) * options['autoencoder_weight']
        if options['perceptual_loss']:
            errEG += torch.mean((netP(netG(netE(images))) - netP(images))**2)
        # Crazy hack: increase autoencoder weight with each epoch
        errEG *= epoch
        errEG.backward()
        ############################

        optimizerE.step()
        optimizerG.step()

        if i % 25 == 0:
            msg = '[{}][{}/{}] D:{:.3f} G:{:.3f} EG:{:.3f}'
            msg = msg.format(
                  epoch, i, len(dataloader),
                  errD.data[0],
                  errG.data[0],
                  errEG.data[0])
            print(msg)
            caption = "Epoch {:04d} iter {:05d}".format(epoch, i)
            reconstructed = netG(netE(images))
            img = torch.cat([images[:12], reconstructed.data[:12], fake.data[:12]])
            filename = "{}/demo_{}.jpg".format(result_dir, int(time.time()))
            imutil.show(img, caption=msg, font_size=8, filename=filename)
    return video_filename


def gen_noise(noise, spherical_noise):
    noise.data.normal_(0, 1)
    if spherical_noise:
        noise = clamp_to_unit_sphere(noise)
    return noise


def clamp_to_unit_sphere(x):
    norm = torch.norm(x, p=2, dim=1)
    norm = norm.expand(1, x.size()[0])
    return torch.mul(x, 1/norm.t())


def train_classifier(networks, optimizers, dataloader, **options):
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

    correct = 0
    total = 0
    
    for i, (images, labels, _) in enumerate(dataloader):
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


def shuffle(a, b, c):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)


def train_active_learning(networks, optimizers, active_points, active_labels, complementary_points, complementary_labels, classifier_name, **options):
    netC = networks[classifier_name]
    netE = networks['encoder']
    netG = networks['generator']
    for net in networks.values():
        net.train()
    # Do not update the generator
    netG.eval()

    optimizerC = optimizers[classifier_name]
    optimizerE = optimizers['encoder']
    result_dir = options['result_dir']
    latent_size = options['latent_size']
    batch_size = options['batch_size']
    use_negative_labels = options['use_complementary_labels']

    is_positive = np.array([1.] * len(active_points) + [0] * len(complementary_points))
    if len(complementary_points) > 0:
        points = np.concatenate([np.array(active_points).squeeze(1), complementary_points.squeeze(1)])
        labels = np.concatenate([active_labels, complementary_labels])
    else:
        points = active_points
        labels = active_labels

    if len(points) == 0:
        print("Warning: no input data available, skipping training")
        return 0
    if len(points) < batch_size:
        print("Warning: not enough data to fill one batch")
        batch_size = len(points) - 1
        print("Setting batch size to {}".format(batch_size))

    def generator(points, labels, is_positive):
        assert len(points) == len(labels)
        while True:
            i = 0
            shuffle(points, labels, is_positive)
            while i < len(points) - batch_size:
                x = torch.FloatTensor(points[i:i+batch_size])
                y = torch.LongTensor(labels[i:i+batch_size])
                is_positive_mask = torch.FloatTensor(is_positive[i:i+batch_size])
                yield x.squeeze(1), y, is_positive_mask
                i += batch_size

    # Train on combined normal and complementary labels
    dataloader = generator(points, labels, is_positive)

    complementary_weight = 1.00
    correct = 0
    total = 0

    batches = (len(active_points) + len(complementary_points)) // batch_size
    print("Training on {} batches".format(batches))
    for i in range(batches):
        latent_points, labels, is_positive_mask = next(dataloader)
        latent_points = Variable(latent_points).cuda()
        labels = Variable(labels).cuda()
        is_positive_mask = Variable(is_positive_mask).cuda()
        is_negative_mask = 1 - is_positive_mask
        negative_count = is_negative_mask.sum().data.cpu().numpy()[0]

        ############################
        # Update C(Z) only
        ############################
        class_predictions = netC(latent_points)
        errPos = masked_nll_loss(class_predictions, labels, is_positive_mask)

        if use_negative_labels and negative_count > 0:
            epsilon = .0001
            y_preds = torch.exp(torch.gather(class_predictions, 1, labels.view(-1, 1)))
            errNeg = -torch.mean(torch.log(1 - y_preds + epsilon) * is_negative_mask)
            errNeg *= complementary_weight / negative_count
        else:
            errNeg = errPos * 0

        errC = errPos + errNeg
        errC.backward()
        optimizerC.step()
        ############################

        _, predicted = class_predictions.max(1)
        correct += sum((predicted.data == labels.data) * (is_positive_mask > 0).data)
        total += sum(is_positive_mask).data[0]

    print('[{}/{}] Pos Loss: {:.3f} Neg Loss: {:.3f} Classifier Accuracy:{:.3f}'.format(
        i, batches, errPos.data[0], errNeg.data[0], float(correct) / total))

    return float(correct) / total


# https://github.com/pytorch/pytorch/issues/563
def masked_nll_loss(logp, y, binary_mask):
    # prepare an (N,C) array of 1s at the locations of ground truth
    ymask = logp.data.new(logp.size()).zero_() # (N,C) all zero
    ymask.scatter_(1, y.data.view(-1,1), 1) # have to make y into shape (N,1) for scatter_ to be happy
    ymask = Variable(ymask)
    logpy = (logp * ymask).sum(1)
    return -(logpy * binary_mask).mean()
