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
from vector import gen_noise, clamp_to_unit_sphere
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

    fixed_noise = Variable(torch.FloatTensor(batch_size, latent_size).normal_(0, 1)).cuda()
    fixed_noise = clamp_to_unit_sphere(fixed_noise)
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
    margin = 100

    for i, (images, class_labels) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(class_labels)

        ############################
        # Generator Updates
        ############################
        if i % discriminator_per_gen == 0:
            netG.zero_grad()
            z = gen_noise(batch_size, latent_size)
            z = Variable(z).cuda()
            gen_images = netG(z)
            
            # Feature Matching: Average of one batch of real vs. generated
            features_real = netD(images, return_features=True)
            features_gen = netD(gen_images, return_features=True)
            fm_loss = torch.mean((features_real.mean(0) - features_gen.mean(0)) ** 2)

            # Pull-away term from https://github.com/kimiyoung/ssl_bad_gan
            nsample = features_gen.size(0)
            denom = features_gen.norm(dim=0).expand_as(features_gen)
            gen_feat_norm = features_gen / denom
            cosine = torch.mm(features_gen, features_gen.t())
            mask = Variable((torch.ones(cosine.size()) - torch.diag(torch.ones(nsample))).cuda())
            pt_loss = torch.sum((cosine * mask) ** 2) / (nsample * (nsample + 1))
            pt_loss /= (1024 * 1024)

            errG = fm_loss + pt_loss * .0001

            errG.backward()
            optimizerG.step()
        ###########################

        ############################
        # Discriminator Updates
        ###########################
        netD.zero_grad()

        # Classify generated examples as the K+1th "open" class
        z = gen_noise(batch_size, latent_size)
        z = Variable(z).cuda()
        fake_images = netG(z).detach()
        fake_logits = netD(fake_images)
        augmented_logits = F.pad(fake_logits, pad=(0,1))
        log_prob_fake = F.log_softmax(augmented_logits, dim=1)
        err_fake = -log_prob_fake[:, -1].mean()
        #err_fake = (log_sum_exp(fake_logits)).mean()
        errD = err_fake * .1
        errD.backward()

        # Classify real examples into the correct K classes
        real_logits = netD(images)
        positive_labels = (labels == 1).type(torch.cuda.FloatTensor)
        augmented_logits = F.pad(real_logits, pad=(0,1))
        augmented_labels = F.pad(positive_labels, pad=(0,1))
        log_likelihood = F.log_softmax(augmented_logits, dim=1) * augmented_labels
        errC = -log_likelihood.mean() * 10
        errC.backward()

        # Classify human-labeled active learning data
        if use_aux_dataset:
            aux_images, aux_labels = aux_dataloader.get_batch()
            aux_images = Variable(aux_images)
            aux_labels = Variable(aux_labels)
            aux_logits = netD(aux_images)
            """
            aux_positive_labels = (aux_labels == 1).type(torch.cuda.FloatTensor)
            aux_negative_labels = (aux_labels == -1).type(torch.cuda.FloatTensor)
            errHingeNegAux = F.softplus(aux_logits) * aux_negative_labels
            errHingePosAux = F.softplus(-aux_logits) * aux_positive_labels
            errNLLAux = -log_softmax(aux_logits, dim=1) * aux_positive_labels
            errHingeNegAux = errHingeNegAux.mean()
            errHingePosAux = errHingePosAux.mean()
            errNLLAux = errNLLAux.mean()
            errCAux = errHingeNegAux + errHingePosAux # + errNLLAux
            import pdb; pdb.set_trace()
            """
            # Use all aux_dataset as negative examples for netD
            errCAux = F.softplus(log_sum_exp(aux_logits)).mean()
            errCAux.backward()

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
            print("fm_loss {:.3f}".format(fm_loss.data[0]))
            print("pt_loss {:.3f}".format(pt_loss.data[0]))
            print("Accuracy {}/{}".format(correct, total))
    return True

