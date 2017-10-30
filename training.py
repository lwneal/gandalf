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
    spherical_noise = type(netE).__name__.endswith('sphere')

    noise = Variable(torch.FloatTensor(batch_size, latent_size).cuda())
    fixed_noise = Variable(torch.FloatTensor(batch_size, latent_size).normal_(0, 1)).cuda()
    if spherical_noise:
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
            D_real_output = netD(images)
            errD_real = D_real_output.mean()
            errD_real.backward(label_one)

            noise.data.normal_(0, 1)
            if spherical_noise:
                noise = clamp_to_unit_sphere(noise)
            fake = netG(noise)
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
        noise.data.normal_(0, 1)
        if spherical_noise:
            noise = clamp_to_unit_sphere(noise)
        fake = netG(noise)
        DG_fake_output = netD(fake)
        errG = DG_fake_output.mean() * options['gan_weight']
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
        # (4) Make reconstructed examples realistic
        # WGAN: minimize D(G(E(x)))
        ############################
        encoded_real = encoded.detach()
        DGE_output = netD(netG(encoded_real))
        errDGE = DGE_output.mean() * options['gan_weight']
        errDGE.backward(label_one)
        ###########################

        ############################
        # (5) Update E(G()) network:
        # Inverse Autoencoder: Minimize Z - E(G(Z))
        ############################
        noise.data.normal_(0, 1)
        if spherical_noise:
            noise = clamp_to_unit_sphere(noise)
        fake = netG(noise)
        reencoded = netE(fake)
        errEG = torch.mean((reencoded - noise) ** 2)
        errEG.backward()
        ############################

        # If joint training is disabled, stop training netE at this point
        if not options['supervised_encoder']:
            optimizerE.step()

        ############################
        # (6) Update C(Z) network:
        # Categorical Cross-Entropy
        ############################
        latent_points = netE(images)
        class_predictions = netC(latent_points)
        errC = nll_loss(class_predictions, labels)
        if not options['attributes_only']:
            errC.backward(retain_graph=True)
        ############################


        ############################
        # (7) Update C(Z) network for attributes:
        # Binary Cross-Entropy
        ############################
        if netA:
            attribute_predictions = netA(latent_points)
            errA = binary_cross_entropy(attribute_predictions, attributes)
            errA.backward()
            optimizerA.step()
        ############################

        if options['supervised_encoder']:
            optimizerE.step()
        optimizerG.step()
        if not options['attributes_only']:
            optimizerC.step()

        # https://discuss.pytorch.org/t/argmax-with-pytorch/1528/2
        _, predicted = class_predictions.max(1)
        correct += sum(predicted.data == labels.data)
        total += len(predicted)

        if netA:
            attr_preds = (attribute_predictions > 0.5) == (attributes > 0.5)
            attr_correct += torch.sum(attr_preds).data.cpu().numpy()[0]
            attr_total += attr_preds.size()[0] * attr_preds.size()[1]

        errD = errD_real + errD_fake
        if i % 25 == 0:
            msg = '[{}][{}/{}] D:{:.3f} G:{:.3f} GP:{:.3f} GE:{:.3f} EG:{:.3f} EC: {:.3f} C_acc:{:.3f} A_acc: {:.3f}'
            msg = msg.format(
                  epoch, i, len(dataloader),
                  errD.data[0],
                  errG.data[0],
                  gradient_penalty.data[0],
                  errGE.data[0],
                  errEG.data[0],
                  errC.data[0],
                  float(correct) / total,
                  float(attr_correct) / attr_total if attr_total > 0 else 0,
                  errDGE.data[0])
            print(msg)
            caption = "Epoch {:04d} iter {:05d}".format(epoch, i)
            demo_gen = netG(fixed_noise)
            imutil.show(demo_gen, video_filename=video_filename, caption=caption, display=False)
        if i % 100 == 0:
            img = torch.cat([images[:12], reconstructed.data[:12], demo_gen.data[:12]])
            filename = "{}/demo_{}.jpg".format(result_dir, int(time.time()))
            imutil.show(img, caption=msg, font_size=8, filename=filename)
    return video_filename


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
    use_negative_labels = options['use_negative_labels']

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

    learning_rate = .1
    correct = 0
    total = 0

    batches = (len(active_points) + len(complementary_points)) // batch_size
    print("Training on {} batches".format(batches))
    for i in range(batches):
        latent_points, labels, is_positive_mask = next(dataloader)
        latent_points = Variable(latent_points).cuda()
        labels = Variable(labels).cuda()
        is_positive_mask = Variable(is_positive_mask).cuda()

        ############################
        # Update E(X) and C(Z) based on positive labels
        # Keep G(Z) constant
        ############################
        generated_images = netG(latent_points).detach()
        class_predictions = netC(latent_points)
        errPos = masked_nll_loss(class_predictions, labels, is_positive_mask)

        if use_negative_labels:
            # A simpler form of complementary label loss: minimize softmax output
            N, K = class_predictions.size()
            y_preds = torch.exp(torch.gather(class_predictions, 1, labels.view(-1,1)))
            errNeg = .001 * learning_rate * (y_preds.squeeze(1) * (1 - is_positive_mask)).sum()
        else:
            errNeg = errPos * 0

        errC = errPos + errNeg
        errC.backward()
        optimizerC.step()
        optimizerE.step()
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
