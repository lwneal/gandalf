import os
import random
import time
import torch
import numpy as np
from torch import autograd
from torch.autograd import Variable
from torch.nn.functional import nll_loss, cross_entropy
import imutil

CF_VIDEO_FRAMES = 48  # Two seconds of video


# Morphs an entire batch of input examples into a given desired_class
def generate_trajectory_batch(networks, dataloader, desired_class=None, **options):
    netG = networks['generator']
    netE = networks['encoder']
    netC = networks['classifier']
    netD = networks['discriminator']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    if desired_class is None:
        desired_class = random.randint(0, dataloader.num_classes - 1)
    print("Morphing input examples to class {}".format(desired_class))

    real_images, labels = dataloader.get_batch()
    real_images = Variable(real_images)

    # We start with vectors in the latent space Z
    z = to_np(netE(real_images))
    z = Variable(torch.FloatTensor(z), requires_grad=True).cuda()
    D_real = netD(real_images).data.cpu().numpy().mean()

    # We want to move them so their classification changes
    target_labels = torch.LongTensor(batch_size)
    target_labels[:] = desired_class
    target_labels = Variable(target_labels).cuda()

    momentum = Variable(torch.zeros(z.size())).cuda()

    # Write all counterfactuals to the trajectories/ subdirectory
    trajectory_id = '{}_{}'.format(dataloader.dsf.name, int(time.time() * 1000))
    video_filename = 'batch-{}-{}.mjpeg'.format(trajectory_id, desired_class)
    video_filename = os.path.join('trajectories', video_filename)
    video_filename = os.path.join(options['result_dir'], video_filename)

    path = os.path.join(options['result_dir'], 'trajectories')
    if not os.path.exists(path):
        print("Creating trajectories directory {}".format(path))
        os.mkdir(path)

    for i in range(200):
        hallucinations = netG(z)
        cf_loss = nll_loss(netC(z), target_labels)
        dc_dz = autograd.grad(cf_loss, z, cf_loss, retain_graph=True)[0]
        momentum -= dc_dz * .01
        z += momentum
        momentum *= .99
        print("Loss: {}".format(cf_loss.data[0]))
        print("Latent point: {}...".format(z[0].data.cpu().numpy()[:5]))
        print("Gradient: {}...".format(dc_dz[0].data.cpu().numpy()[:5]))
        print("Momentum: {}...".format(momentum[0].data.cpu().numpy()[:5]))
        classes = to_np(netC(z).max(1)[1])
        print("Class: {}...".format(classes))

        D_halluc = netD(hallucinations).data.cpu().numpy().mean()

        caption = "DR {:.04f}  DG {:.04f}".format(D_real, D_halluc)
        imutil.show(hallucinations,
                video_filename=video_filename,
                caption=caption,
                font_size=12,
                display=False)
    imutil.encode_video(video_filename)
    return to_np(z)


def to_torch(z, requires_grad=False):
    return Variable(torch.FloatTensor(z), requires_grad=requires_grad).cuda()


def to_np(z):
    return z.data.cpu().numpy()


def generate_trajectory_active(networks, dataloader, **options):
    netG = networks['generator']
    netE = networks['encoder']
    netC = networks['classifier']
    netD = networks['discriminator']
    result_dir = options['result_dir']
    image_size = options['image_size']
    latent_size = options['latent_size']


    real_image, label = dataloader.get_batch()
    real_image = Variable(real_image)

    start_class = label.cpu().numpy()[0]
    target_class = random.randint(0, dataloader.num_classes - 1)

    print("Morphing input example from class {} to class {}".format(start_class, target_class))

    # We start with vectors in the latent space Z
    z = to_np(netE(real_image))
    z = Variable(torch.FloatTensor(z), requires_grad=True).cuda()
    D_real = netD(real_image).data.cpu().numpy().mean()

    # We want to move them so their classification changes
    target_label = torch.LongTensor(1)
    target_label[:] = target_class
    target_label = Variable(target_label).cuda()

    momentum = Variable(torch.zeros(z.size())).cuda()

    # Write all counterfactuals to the trajectories/ subdirectory
    trajectory_id = '{}_{}'.format(dataloader.dsf.name, int(time.time() * 1000))
    video_filename = 'active-{}-{}-{}.mjpeg'.format(trajectory_id, start_class, target_class)
    video_filename = os.path.join('trajectories', video_filename)
    video_filename = os.path.join(options['result_dir'], video_filename)

    trajectory_filename = video_filename.replace('.mjpeg', '.npy')

    path = os.path.join(options['result_dir'], 'trajectories')
    if not os.path.exists(path):
        print("Creating trajectories directory {}".format(path))
        os.mkdir(path)

    z_trajectory = []

    for i in range(200):
        hallucinations = netG(z)
        cf_loss = nll_loss(netC(z), target_label)
        dc_dz = autograd.grad(cf_loss, z, cf_loss, retain_graph=True)[0]
        momentum -= dc_dz * .0001
        z += momentum
        momentum *= .50
        print("Loss: {}".format(cf_loss.data[0]))
        print("Latent point: {}...".format(z[0].data.cpu().numpy()[:5]))
        print("Gradient: {}...".format(dc_dz[0].data.cpu().numpy()[:5]))
        print("Momentum: {}...".format(momentum[0].data.cpu().numpy()[:5]))
        classes = to_np(netC(z).max(1)[1])
        print("Class: {}...".format(classes))

        D_halluc = netD(hallucinations).data.cpu().numpy().mean()

        caption = "DR {:.04f}  DG {:.04f}".format(D_real, D_halluc)
        imutil.show(hallucinations,
                video_filename=video_filename,
                caption=caption,
                font_size=12,
                resize_to=(512,512),
                display=False)

        z_trajectory.append(to_np(z))

    print("Encoding video...")
    imutil.encode_video(video_filename)

    print("Saving trajectory")
    np.save(trajectory_filename, np.array(z_trajectory))

    return to_np(z)
