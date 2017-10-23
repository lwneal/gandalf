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
    netA = networks.get('attribute')
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    if desired_class is None:
        desired_class = random.randint(0, dataloader.num_classes - 1)
    desired_class_name = dataloader.lab_conv.labels[desired_class]
    print("Morphing input examples to class {}".format(desired_class_name))

    real_images, labels, attributes = dataloader.get_batch(required_class=options['start_class'])
    real_images = Variable(real_images)

    # We start with vectors in the latent space Z
    z = to_np(netE(real_images))
    z = Variable(torch.FloatTensor(z), requires_grad=True).cuda()
    original_z = z.clone()
    D_real = netD(real_images).data.cpu().numpy().mean()
    if netA:
        original_preds = netA(z)

    # We want to move them so their classification changes
    target_labels = torch.LongTensor(batch_size)
    target_labels[:] = desired_class
    target_labels = Variable(target_labels).cuda()

    momentum = Variable(torch.zeros(z.size())).cuda()

    # Write all counterfactuals to the trajectories/ subdirectory
    trajectory_id = '{}_{}'.format(dataloader.dsf.name, int(time.time() * 1000))
    video_filename = 'batch-{}-{}.mjpeg'.format(trajectory_id, desired_class_name)
    video_filename = os.path.join('trajectories', video_filename)
    video_filename = os.path.join(options['result_dir'], video_filename)

    path = os.path.join(options['result_dir'], 'trajectories')
    if not os.path.exists(path):
        print("Creating trajectories directory {}".format(path))
        os.mkdir(path)

    # First frame: originals, for comparison
    imutil.show(real_images,
            video_filename=video_filename,
            caption="Original",
            font_size=12,
            display=False)

    z_trajectory = []

    # TODO: Refactor all of this into a function, have it return the trajectory
    # Then do processing on the trajectory
    for i in range(90):
        target_attr = None
        if options['zero_attribute']:
            target_attr = options['zero_attribute']
            target_attr_value = 0
        if options['one_attribute']:
            target_attr = options['one_attribute']
            target_attr_value = 1

        # Inner loop: Take several steps of gradient descent
        for _ in range(10):
            if target_attr:
                # Attribute-based counterfactual
                attr_idx = dataloader.attr_conv.idx[target_attr]
                preds = netA(z)
                target_attrs = preds.clone()
                target_attrs[:,attr_idx] = 0
                yhat = preds[:,attr_idx]
                epsilon = .01
                cf_loss = -target_attr_value * torch.log(epsilon + yhat) - (1 - target_attr_value) * torch.log(epsilon + 1 - yhat)
                cf_loss = torch.mean(cf_loss)
            else:
                # Class-based counterfactual
                cf_loss = nll_loss(netC(z), target_labels)

            # Distance in latent space from original point
            cf_loss += .002 * torch.sum((z - original_z) ** 2)

            dc_dz = autograd.grad(cf_loss, z, cf_loss, retain_graph=True)[0]
            momentum -= dc_dz * .01
            z += momentum
            momentum *= .5

        z_trajectory.append(to_np(z))

        print("Loss: {}".format(cf_loss.data[0]))
        print("Latent point: {}...".format(z[0].data.cpu().numpy()[:5]))
        print("Gradient: {}...".format(dc_dz[0].data.cpu().numpy()[:5]))
        print("Momentum: {}...".format(momentum[0].data.cpu().numpy()[:5]))

        classes = to_np(netC(z).max(1)[1])
        print("Class: {}...".format(classes))

        hallucinations = netG(z)
        D_halluc = netD(hallucinations).data.cpu().numpy().mean()
        
        caption = "DR {:.04f}  DG {:.04f}".format(D_real, D_halluc)
        imutil.show(hallucinations,
                video_filename=video_filename,
                caption=caption,
                font_size=12,
                display=i % 25 == 0)

    # A .mp4 video for normal visualization
    imutil.encode_video(video_filename)

    # A still frame with no caption, used for the Batch Labeling UI
    imutil.show(hallucinations, filename=video_filename.replace('.mjpeg', '.jpg'))

    trajectory_filename = video_filename.replace('.mjpeg', '.npy')
    print("Saving trajectory {}".format(trajectory_filename))
    np.save(trajectory_filename, np.array(z_trajectory))
    
    if netA:
        print("Original Attributes:")
        print(dataloader.attr_conv.attributes)
        before = torch.sum(original_preds, 0)
        print("Counterfactual Attributes:")
        after = torch.sum(netA(z), 0)
        print(after - before)

    return to_np(z)


def to_torch(z, requires_grad=False):
    return Variable(torch.FloatTensor(z), requires_grad=requires_grad).cuda()


def to_np(z):
    return z.data.cpu().numpy()


def generate_trajectory_active(networks, dataloader, strategy='random', **options):
    netG = networks['generator']
    netE = networks['encoder']
    netC = networks['classifier']
    netD = networks['discriminator']
    result_dir = options['result_dir']
    image_size = options['image_size']
    latent_size = options['latent_size']
    output_samples = options['counterfactual_frame_count']
    speed = options['speed']
    momentum_mu = options['momentum_mu']

    if strategy == 'random':
        real_image, label, attributes = dataloader.get_batch()
        real_image = Variable(real_image)

        start_class = label.cpu().numpy()[0]
        target_class = random.randint(0, dataloader.num_classes - 2)
        if start_class <= target_class:
            target_class += 1
    elif strategy == 'uncertainty':
        print("Performing uncertainty sampling with dataloader {}".format(dataloader))
        # todo: run the classifier on every? point, and find one with minimal top-1 score
        # Then decide on which direction to take it
        real_image, label, attributes = dataloader.get_batch()
        real_image = Variable(real_image)
        print("TODO: Run through a pool of unlabeled examples, apply classifier to each one")
        print("TODO: Pick the highest-uncertainty unlabeled example")

        start_class = label.cpu().numpy()[0]
        target_class = random.randint(0, dataloader.num_classes - 2)
        if start_class <= target_class:
            target_class += 1
    else:
        raise ValueError("Unknown sampling strategy {}".format(strategy))

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
    start_class_name = dataloader.lab_conv.labels[start_class]
    target_class_name = dataloader.lab_conv.labels[target_class]
    video_filename = 'active-{}-{}-{}.mjpeg'.format(trajectory_id, start_class_name, target_class_name)
    video_filename = os.path.join('trajectories', video_filename)
    video_filename = os.path.join(options['result_dir'], video_filename)

    trajectory_filename = video_filename.replace('.mjpeg', '.npy')

    path = os.path.join(options['result_dir'], 'trajectories')
    if not os.path.exists(path):
        print("Creating trajectories directory {}".format(path))
        os.mkdir(path)

    # Generate z_trajectory
    z_trajectory = []
    MIN_ITERS = 30
    MAX_ITERS = 2000
    for i in range(MAX_ITERS):
        cf_loss = nll_loss(netC(z), target_label)
        dc_dz = autograd.grad(cf_loss, z, cf_loss, retain_graph=True)[0]
        momentum -= dc_dz * speed
        z += momentum
        momentum *= momentum_mu
        preds = netC(z)
        predicted_class = to_np(preds.max(1)[1])[0]
        pred_confidence = np.exp(to_np(preds.max(1)[0])[0])
        z_trajectory.append(to_np(z))
        if len(z_trajectory) >= MIN_ITERS:
            if predicted_class == target_class and pred_confidence > .99:
                break
            if np.linalg.norm(z_trajectory[-1] - z_trajectory[-2]) < .0001:
                break
        if i % 100 == 0:
            print("Iter {} Loss: {}".format(i, cf_loss.data[0]))
            print("Latent point: {}...".format(z[0].data.cpu().numpy()[:5]))
            print("Gradient: {}...".format(dc_dz[0].data.cpu().numpy()[:5]))
            print("Momentum: {}...".format(momentum[0].data.cpu().numpy()[:5]))
            print("Class: {} ({:.3f} confidence)...".format(predicted_class, pred_confidence))
    predicted_class_name = dataloader.lab_conv.labels[predicted_class]
    print("Class: {} ({:.3f} confidence)...".format(predicted_class_name, pred_confidence))

    # Normalize z_trajectory and turn it into a video
    sampled_trajectory = sample_trajectory(z_trajectory, output_samples=output_samples)
    for z in sampled_trajectory:
        z = Variable(torch.FloatTensor(z)).cuda()
        hallucinations = netG(z)
        D_halluc = netD(hallucinations).data.cpu().numpy().mean()
        preds = netC(z)
        predicted_class = to_np(preds.max(1)[1])[0]
        predicted_class_name = dataloader.lab_conv.labels[predicted_class]
        pred_confidence = np.exp(to_np(preds.max(1)[0])[0])
        caption = "Class: {} (confidence {:.3f})".format(predicted_class_name, pred_confidence)
        imutil.show(hallucinations,
                video_filename=video_filename,
                caption=caption,
                font_size=12,
                resize_to=(512,512),
                display=False)

    imutil.encode_video(video_filename)

    print("Saving trajectory length {} to {}".format(len(sampled_trajectory), trajectory_filename))
    np.save(trajectory_filename, np.array(sampled_trajectory))

    return to_np(z)


def sample_trajectory(zt, output_samples=30):
    distances = np.array([np.linalg.norm(zt[i+1] - zt[i]) for i in range(len(zt) - 1)])
    total_distance = sum(distances)
    distance_per_sample = total_distance / output_samples 

    samples = []
    cumulative_distance = 0
    for i in range(len(distances)):
        if len(samples) * distance_per_sample <= cumulative_distance:
            samples.append(zt[i])
        cumulative_distance += distances[i]
    print("{} samples, expecting {}".format(len(samples), output_samples))
    assert len(samples) == output_samples
    return samples
