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


# Morphs an entire batch of input examples into a given target_class
def generate_trajectory_batch(networks, dataloader, target_class=None, **options):
    netG = networks['generator']
    netE = networks['encoder']
    netC = networks['classifier']
    netD = networks['discriminator']
    netA = networks.get('attribute')
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']
    speed = options['speed']
    momentum_mu = options['momentum_mu']
    max_iters = options['counterfactual_max_iters']

    if target_class is None:
        target_class = random.randint(0, dataloader.num_classes - 1)
    target_class_name = dataloader.lab_conv.labels[target_class]
    print("Morphing input examples to class {}".format(target_class_name))

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
    target_labels[:] = target_class
    target_labels = Variable(target_labels).cuda()

    momentum = Variable(torch.zeros(z.size())).cuda()

    # Write all counterfactuals to the trajectories/ subdirectory
    trajectory_id = '{}_{}'.format(dataloader.dsf.name, int(time.time() * 1000))
    video_filename = 'batch-{}-{}.mjpeg'.format(trajectory_id, target_class_name)
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
            cf_loss += .00001 * torch.sum((z - original_z) ** 2)

            dc_dz = autograd.grad(cf_loss, z, cf_loss, retain_graph=True)[0]
            momentum -= dc_dz * speed
            z += momentum
            momentum *= momentum_mu

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
    max_iters = options['counterfactual_max_iters']
    MAX_CANDIDATE_POINTS = 1000  # speed tradeoff


    if strategy == 'uncertainty':
        print("Performing uncertainty sampling with dataloader {}".format(dataloader))
        images = []
        candidate_points = []
        scores = []
        for i, (img, label, attr) in enumerate(dataloader):
            images.append(img)
            z = netE(Variable(img))
            certainty = torch.exp(netC(netE(Variable(img)))).max()
            candidate_points.append(z)
            scores.append(float(certainty.data[0]))
            if i > MAX_CANDIDATE_POINTS:
                break
        idx = np.argmin(scores)
        real_image = Variable(images[idx])
        z_val = to_np(candidate_points[idx])
        print("Uncertainty sampling picking point {} with certainty {}".format(idx, scores[idx]))
        preds = to_np(torch.exp(netC(candidate_points[idx])))
        top1_idx = np.argmax(preds)
        top1_conf = np.max(preds)
        preds[0][top1_idx] = 0
        top2_idx = np.argmax(preds)
        top2_conf = np.max(preds)
        print("Most likely class: {} (score {})".format(top1_idx, top1_conf))
        print("Next likely class: {} (score {})".format(top2_idx, top2_conf))
        start_class = top1_idx
        target_class = top2_idx
    else:
        real_image, label, attributes = dataloader.get_batch(required_class=options['start_class'])
        real_image = Variable(real_image)
        start_class = label.cpu().numpy()[0]
        if options['target_class']:
            target_class = dataloader.lab_conv.idx[options['target_class']]
        else:
            target_class = random.randint(0, dataloader.num_classes - 2)
            if start_class <= target_class:
                target_class += 1
        print("Morphing input example from class {} to class {}".format(start_class, target_class))

        # We start with vectors in the latent space Z
        z_val = to_np(netE(real_image))

    z = Variable(torch.FloatTensor(z_val), requires_grad=True).cuda()
    original_z = z.clone()
    D_real = netD(real_image).data.cpu().numpy().mean()

    target_label = torch.LongTensor(1)
    target_label[:] = int(target_class)
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
    for i in range(max_iters):
        cf_loss = nll_loss(netC(z), target_label)

        # Distance in latent space from original point
        cf_loss += .0001 * torch.sum((z - original_z) ** 2)

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

    def output_frame(hallucinations, caption, idx):
        # Always write the video
        imutil.show(hallucinations*255.,
                video_filename=video_filename,
                caption=caption,
                font_size=12,
                resize_to=(512,512),
                normalize_color=False,
                display=False)
        if options['write_jpgs']:
            jpg_dir = os.path.join(options['result_dir'], 'trajectory_jpgs')
            if not os.path.exists(jpg_dir):
                os.mkdir(jpg_dir)
            jpg_filename = 'active-{}-{}-{}-{:04d}.jpg'.format(trajectory_id, start_class_name, target_class_name, idx)
            jpg_filename = os.path.join(options['result_dir'], 'trajectory_jpgs', jpg_filename)
            imutil.show(hallucinations*255., filename=jpg_filename, resize_to=(512,512), display=False, normalize_color=False)

    # Normalize z_trajectory and turn it into a video
    for i in range(4):
        output_frame(real_image, caption="Original", idx=0)
    sampled_trajectory = sample_trajectory(z_trajectory, output_samples=output_samples)
    for i, z in enumerate(sampled_trajectory):
        z = Variable(torch.FloatTensor(z)).cuda()
        hallucinations = netG(z)
        D_halluc = netD(hallucinations).data.cpu().numpy().mean()
        preds = netC(z)
        predicted_class = to_np(preds.max(1)[1])[0]
        predicted_class_name = dataloader.lab_conv.labels[predicted_class]
        pred_confidence = np.exp(to_np(preds.max(1)[0])[0])
        caption = "Class: {} (confidence {:.3f})".format(predicted_class_name, pred_confidence)
        output_frame(hallucinations, caption, idx=i+1)


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
    return samples
