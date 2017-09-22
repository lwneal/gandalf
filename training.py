import time
import torch
from torch.autograd import Variable
from gradient_penalty import calc_gradient_penalty
from torch.nn.functional import nll_loss
import imutil


def train_adversarial_autoencoder(models, optimizers, dataloader, epoch=None, **params):
    netD = models['discriminator']
    netG = models['generator']
    netE = models['encoder']
    netC = models['classifier']
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    optimizerE = optimizers['encoder']
    optimizerC = optimizers['classifier']
    epochs = params['epochs']
    resultDir = params['resultDir']
    batchSize = params['batchSize']
    imageSize = params['imageSize']
    latentSize = params['latentSize']

    real_input = torch.FloatTensor(batchSize, 3, imageSize, imageSize).cuda()
    noise = torch.FloatTensor(batchSize, latentSize).cuda()
    fixed_noise = Variable(torch.FloatTensor(batchSize, latentSize).normal_(0, 1)).cuda()
    label_one = torch.FloatTensor(batchSize).cuda().fill_(1)
    label_zero = torch.FloatTensor(batchSize).cuda().fill_(0)
    label_minus_one = torch.FloatTensor(batchSize).cuda().fill_(-1)
    correct = 0
    total = 0

    for i, data in enumerate(dataloader):
        images, labels = data
        ############################
        # (1) Update D network
        # WGAN: maximize D(G(z)) - D(x)
        ###########################
        for _ in range(5):
            netD.zero_grad()
            D_real_output = netD(Variable(images))
            errD_real = D_real_output.mean()
            errD_real.backward(label_one)

            noise.normal_(0, 1)
            fake = netG(Variable(noise))
            fake = fake.detach()
            D_fake_output = netD(fake)
            errD_fake = D_fake_output.mean()
            errD_fake.backward(label_minus_one)

            gradient_penalty = calc_gradient_penalty(netD, images, fake.data)
            gradient_penalty.backward()

            optimizerD.step()
        ###########################

        ############################
        # (2) Update G network:
        # WGAN: minimize D(G(z))
        ############################
        netG.zero_grad()
        noise.normal_(0, 1)
        fake = netG(Variable(noise))
        DG_fake_output = netD(fake)
        errG = DG_fake_output.mean()
        errG.backward(label_one)
        optimizerG.step()
        ###########################

        ############################
        # (3) Update G(E()) network:
        # Autoencoder: Minimize X - G(E(X))
        ############################
        netE.zero_grad()
        netG.zero_grad()
        encoded = netE(Variable(images))
        reconstructed = netG(encoded)
        errGE = torch.mean(torch.abs(reconstructed - Variable(images)))
        errGE.backward()
        ############################
        # (4) Update E(G()) network:
        # Inverse Autoencoder: Minimize Z - E(G(Z))
        ############################
        noise.normal_(0, 1)
        fake = netG(Variable(noise))
        reencoded = netE(fake)
        errEG = torch.mean((reencoded - Variable(noise)) ** 2)
        errEG.backward()
        optimizerE.step()
        optimizerG.step()
        ############################

        ############################
        # (5) Update C(Z) network:
        # Categorical Cross-Entropy
        ############################
        netE.zero_grad()
        netC.zero_grad()
        latent_points = netE(Variable(images))
        class_predictions = netC(latent_points)
        errC = nll_loss(class_predictions, Variable(labels))
        errC.backward()
        optimizerE.step()
        optimizerC.step()
        ############################

        # https://discuss.pytorch.org/t/argmax-with-pytorch/1528/2
        _, predicted = preds.max(1)
        correct += sum(predicted.data == labels)
        total += len(predicted)

        errD = errD_real + errD_fake
        if i % 25 == 0:
            msg = '[{}/{}][{}/{}] D:{:.3f} G:{:.3f} GP:{:.3f} GE:{:.3f} EG:{:.3f} C_acc:{:.3f}'
            msg = msg.format(
                  epoch, epochs, i, len(dataloader),
                  errD.data[0],
                  errG.data[0],
                  gradient_penalty.data[0],
                  errGE.data[0],
                  errEG.data[0],
                  float(correct) / total)
            print(msg)
            video_filename = "{}/generated.mjpeg".format(resultDir)
            caption = "Epoch {:02d} iter {:05d}".format(epoch, i)
            demo_gen = netG(fixed_noise)
            imutil.show(demo_gen, video_filename=video_filename, caption=caption, display=False)
        if i % 100 == 0:
            img = torch.cat([images[:12], reconstructed.data[:12], demo_gen.data[:12]])
            filename = "{}/demo_{}.jpg".format(resultDir, int(time.time()))
            imutil.show(img, caption=msg, font_size=8, filename=filename)
