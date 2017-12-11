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
    demo_images, _, _ = next(d for d in dataloader)

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
            errD *= options['gan_weight']
            errD.backward()
            optimizerD.step()
        ###########################

        # WGAN-GP
        netD.zero_grad()
        errGP = calc_gradient_penalty(netD, images.data, fake_images.data)
        errGP.backward()
        optimizerD.step()

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
        if options['perceptual_loss']:
            errEG += torch.mean((netP(regenerated) - netP(hallucination))**2)
        errEG *= options['autoencoder_weight']
        errEG.backward()
        ############################
        optimizerG.step()
        optimizerE.step()

        ############################
        # Train Classifier
        ############################
        if options['supervised_encoder']:
            netE.zero_grad()
        netC.zero_grad()
        preds = netC(netE(images))
        errC = nll_loss(preds, labels)
        errC.backward()
        optimizerC.step()
        if options['supervised_encoder']:
            optimizerE.step()

        confidence, pred_idx = preds.max(1)
        correct += sum(pred_idx == labels).data.cpu().numpy()[0]
        total += len(labels)
        ############################

        if i % 100 == 0:
            msg = '[{}][{}/{}] D:{:.3f} G:{:.3f} EG:{:.3f} EC: {:.3f} Acc. {:.3f}'
            msg = msg.format(
                  epoch, i, len(dataloader),
                  errD.data[0],
                  errG.data[0],
                  errEG.data[0],
                  errC.data[0],
                  correct / max(total, 1))
            print(msg)

            caption = "Epoch {:04d} iter {:05d}".format(epoch, i)
            reconstructed = netG(netE(Variable(demo_images)))
            demo_fakes = netG(fixed_noise)
            img = torch.cat([demo_images[:12], reconstructed.data[:12], demo_fakes.data[:12]])
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
