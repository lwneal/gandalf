import time
import torch
import torch.nn as nn
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
        netP = nn.Sequential(*list(vgg16.features.children())[:5])
        netP.cuda()

    real_input = torch.FloatTensor(batch_size, 3, image_size, image_size).cuda()
    noise = torch.FloatTensor(batch_size, latent_size).cuda()
    fixed_noise = Variable(torch.FloatTensor(batch_size, latent_size).normal_(0, 1)).cuda()
    label_one = torch.FloatTensor(batch_size).cuda().fill_(1)
    label_zero = torch.FloatTensor(batch_size).cuda().fill_(0)
    label_minus_one = torch.FloatTensor(batch_size).cuda().fill_(-1)
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images)
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
        errGE = torch.mean(torch.abs(reconstructed - images))
        if options['perceptual_loss']:
            errGE += torch.mean(torch.abs(netP(reconstructed) - netP(images)))
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
        errC = nll_loss(class_predictions, Variable(labels))
        errC.backward()
        ############################

        if not options['joint_encoder_classifier']:
            optimizerE.step()
        optimizerG.step()
        optimizerC.step()

        # https://discuss.pytorch.org/t/argmax-with-pytorch/1528/2
        _, predicted = class_predictions.max(1)
        correct += sum(predicted.data == labels)
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
            caption = "Epoch {:02d} iter {:05d}".format(epoch, i)
            demo_gen = netG(fixed_noise)
            imutil.show(demo_gen, video_filename=video_filename, caption=caption, display=False)
        if i % 100 == 0:
            img = torch.cat([images[:12], reconstructed.data[:12], demo_gen.data[:12]])
            filename = "{}/demo_{}.jpg".format(result_dir, int(time.time()))
            imutil.show(img, caption=msg, font_size=8, filename=filename)
