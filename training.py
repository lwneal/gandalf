import torch
from torch.autograd import Variable
from gradient_penalty import calc_gradient_penalty
import imutil


def train_adversarial_autoencoder(models, optimizers, dataloader, epoch=None, **params):
    netD = models['discriminator']
    netG = models['generator']
    netE = models['encoder']
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    optimizerE = optimizers['encoder']
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

    for i, img_batch in enumerate(dataloader):
        ############################
        # (1) Update D network
        # WGAN: maximize D(G(z)) - D(x)
        ###########################
        for _ in range(5):
            netD.zero_grad()
            D_real_output = netD(Variable(img_batch))
            errD_real = D_real_output.mean()
            errD_real.backward(label_one)

            noise.normal_(0, 1)
            fake = netG(Variable(noise))
            fake = fake.detach()
            D_fake_output = netD(fake)
            errD_fake = D_fake_output.mean()
            errD_fake.backward(label_minus_one)

            gradient_penalty = calc_gradient_penalty(netD, img_batch, fake.data)
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
        encoded = netE(Variable(img_batch))
        reconstructed = netG(encoded)
        errGE = torch.mean(torch.abs(reconstructed - Variable(img_batch)))
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

        errD = errD_real + errD_fake
        if i % 25 == 0:
            print('[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} Loss_GP: {:.4f}  Loss_GE: {:.4f} Loss_EG {:.4f}'.format(
                  epoch, epochs, i, len(dataloader),
                  errD.data[0],
                  errG.data[0],
                  gradient_penalty.data[0],
                  errGE.data[0],
                  errEG.data[0]))
            video_filename = "{}/generated.mjpeg".format(resultDir)
            caption = "Epoch {}".format(epoch)
            demo_img = netG(fixed_noise)
            imutil.show(demo_img, video_filename=video_filename, caption=caption, display=False)
        if i % 100 == 0:
            imutil.show(torch.cat([img_batch[:12], reconstructed.data[:12], demo_img.data[:12]]))
