import os
import random
import time
import torch
import numpy as np
from torch import autograd
from torch.autograd import Variable
from torch.nn.functional import softmax, sigmoid, log_softmax
from torch.nn.functional import nll_loss, cross_entropy
import imutil
from imutil import VideoMaker


def to_torch(z, requires_grad=False):
    return Variable(torch.FloatTensor(z), requires_grad=requires_grad).cuda()


def to_np(z):
    return z.data.cpu().numpy()


# Just picks ground-truth examples with the right labels
def generate_grid(networks, dataloader, **options):
    result_dir = options['result_dir']
    result_dir = options['result_dir']

    # Generate a K by K square grid of examples, one column per class
    K = dataloader.num_classes
    images = [[] for _ in range(N)]
    for img_batch, label_batch, _ in dataloader:
        for img, label in zip(img_batch, label_batch):
            if label < N and len(images[label]) < N:
                images[label].append(img.cpu().numpy())
        if all(len(images[i]) == N for i in range(N)):
            break

    flat = []
    for i in range(N):
        for j in range(N):
            flat.append(images[j][i])
    images = flat

    images = np.array(images).transpose((0,2,3,1))
    start_class = 0
    video_filename = make_video_filename(result_dir, dataloader, start_class, start_class, label_type='grid')
    # Save the images in npy format to re-load as training data
    trajectory_filename = video_filename.replace('.mjpeg', '.npy')
    np.save(trajectory_filename, images)
    # Save the images in jpg format to display to the user
    imutil.show(images, filename=video_filename.replace('.mjpeg', '.jpg'))


# Generates 'counterfactual' images for each class, by gradient descent of the class
def generate_comparison(networks, dataloader, **options):
    for net in networks:
        networks[net].eval()
    netG = networks['generator']
    netD = networks['discriminator']
    result_dir = options['result_dir']
    image_size = options['image_size']
    latent_size = options['latent_size']
    output_frame_count = options['counterfactual_frame_count']
    speed = options['speed']
    momentum_mu = options['momentum_mu']
    max_iters = options['counterfactual_max_iters']
    result_dir = options['result_dir']

    K = dataloader.num_classes

    batches = []
    for class_idx in range(K):
        img_batch = generate_images_for_class(networks, dataloader, class_idx, **options)
        batches.append(img_batch)

    images = []
    for i in range(K):
        for batch in batches:
            images.append(batch[i])

    images = np.array(images).transpose((0,2,3,1))
    dummy_class = 0
    video_filename = make_video_filename(result_dir, dataloader, dummy_class, dummy_class, label_type='grid')

    # Save the images in npy format to re-load as training data
    trajectory_filename = video_filename.replace('.mjpeg', '.npy')
    np.save(trajectory_filename, images)

    # Save the images in jpg format to display to the user
    imutil.show(images, filename=video_filename.replace('.mjpeg', '.jpg'))


def generate_images_for_class(networks, dataloader, class_idx, **options):
    netG = networks['generator']
    netD = networks['discriminator']
    result_dir = options['result_dir']
    image_size = options['image_size']
    latent_size = options['latent_size']
    output_frame_count = options['counterfactual_frame_count']
    speed = options['speed']
    momentum_mu = options['momentum_mu']
    max_iters = options['counterfactual_max_iters']
    result_dir = options['result_dir']

    # Start with K random points
    K = dataloader.num_classes
    z = gen_noise(K, latent_size)
    z = Variable(z, requires_grad=True).cuda()

    # Move them so their labels match target_label
    target_label = torch.LongTensor(K)
    target_label[:] = class_idx
    target_label = Variable(target_label).cuda()

    maxval = None  # So speed=.1 works for any classifier

    for i in range(max_iters):
        images = netG(z)
        net_y = netD(images)
        if maxval is None:
            maxval = net_y.max()
        net_y /= maxval
        preds = softmax(net_y)

        pred_classes = to_np(preds.max(1)[1])
        predicted_class = pred_classes[0]
        pred_confidences = to_np(preds.max(1)[0])
        pred_confidence = pred_confidences[0]
        predicted_class_name = dataloader.lab_conv.labels[predicted_class]
        print("Class: {} ({:.3f} confidence). Target class {}".format(
            predicted_class_name, pred_confidence, class_idx))

        cf_loss = nll_loss(log_softmax(net_y), target_label)

        dc_dz = autograd.grad(cf_loss, z, cf_loss, retain_graph=True)[0]
        z -= dc_dz * speed
        z /= torch.sqrt(torch.mul(z, z).sum())
        if all(pred_classes == class_idx) and all(pred_confidences > 0.99):
            break
    return images.data.cpu().numpy()
    
    
def gen_noise(K, latent_size):
    noise = torch.zeros((K, latent_size))
    noise.normal_(0, 1)
    noise = clamp_to_unit_sphere(noise)
    return noise


def clamp_to_unit_sphere(x):
    norm = torch.norm(x, p=2, dim=1)
    norm = norm.expand(1, x.size()[0])
    return torch.mul(x, 1/norm.t())


# Trajectories are written to result_dir/trajectories/
def make_video_filename(result_dir, dataloader, start_class, target_class, label_type='active'):
    trajectory_id = '{}_{}'.format(dataloader.dsf.name, int(time.time() * 1000))
    start_class_name = dataloader.lab_conv.labels[start_class]
    target_class_name = dataloader.lab_conv.labels[target_class]
    video_filename = '{}-{}-{}-{}.mjpeg'.format(label_type, trajectory_id, start_class_name, target_class_name)
    video_filename = os.path.join('trajectories', video_filename)
    video_filename = os.path.join(result_dir, video_filename)
    path = os.path.join(result_dir, 'trajectories')
    if not os.path.exists(path):
        print("Creating trajectories directory {}".format(path))
        os.mkdir(path)
    return video_filename
