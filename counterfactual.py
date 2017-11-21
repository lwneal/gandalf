import os
import random
import time
import torch
import numpy as np
from torch import autograd
from torch.autograd import Variable
from torch.nn.functional import nll_loss, cross_entropy
import imutil
from imutil import VideoMaker


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
    output_frame_count = options['counterfactual_frame_count']
    speed = options['speed']
    momentum_mu = options['momentum_mu']
    max_iters = options['counterfactual_max_iters']
    result_dir = options['result_dir']

    if strategy == 'random':
        # Start with a random example, move it to a random class
        most_likely_class, least_likely_class, start_score, start_img = select_uncertain_example(dataloader, netE, netC, pool_size=1)
        start_class = most_likely_class
        target_class = random_target_class(dataloader, start_class)
    elif strategy == 'random-nearest':
        # Start with a random example, move it to the nearest decision boundary
        most_likely_class, least_likely_class, start_score, start_img = select_uncertain_example(dataloader, netE, netC, pool_size=1)
        start_class = most_likely_class
        target_class = most_likely_class
    elif strategy == 'uncertainty-random':
        # Start with an uncertain example, move it to a random class
        most_likely_class, least_likely_class, start_score, start_img = select_uncertain_example(dataloader, netE, netC)
        start_class = most_likely_class
        target_class = random_target_class(dataloader, start_class)
    elif strategy == 'uncertainty-nearest':
        # Start with an uncertain example, move it to the nearest class
        most_likely_class, least_likely_class, start_score, start_img = select_uncertain_example(dataloader, netE, netC)
        start_class = most_likely_class
        target_class = most_likely_class
    elif strategy == 'certainty-random':
        # Start with an easy-to-classify sample, move it to the least likely class
        most_likely_class, least_likely_class, start_score, start_img = select_uncertain_example(dataloader, netE, netC, reverse=True)
        start_class = most_likely_class
        target_class = random_target_class(dataloader, start_class)
    elif strategy == 'certainty-furthest':
        # Start with an easy-to-classify sample, move it to the least likely class
        most_likely_class, least_likely_class, start_score, start_img = select_uncertain_example(dataloader, netE, netC, reverse=True)
        start_class = most_likely_class
        target_class = least_likely_class
    elif strategy == 'random-interpolate':
        start_class, _, _, start_img = select_uncertain_example(dataloader, netE, netC, pool_size=1)
        target_class, _, _, target_img = select_uncertain_example(dataloader, netE, netC, pool_size=1)
    else:
        raise ValueError("Unknown strategy")

    if strategy == 'random-interpolate':
        z_trajectory = [to_np(netE(Variable(start_img))), to_np(netE(Variable(target_img)))]
    else:
        # Generate a path in latent space from start_img to a known classification
        z = netE(Variable(start_img))
        z_trajectory = generate_z_trajectory(z, target_class, netC, dataloader, speed, momentum_mu, max_iters=max_iters)

    sampled_trajectory = sample_trajectory(z_trajectory, output_frame_count)

    # Save the trajectory in .npy to later load
    video_filename = make_video_filename(result_dir, dataloader, start_class, target_class)
    trajectory_filename = video_filename.replace('.mjpeg', '.npy')
    print("Saving trajectory length {} to {}".format(len(sampled_trajectory), trajectory_filename))
    np.save(trajectory_filename, np.array(sampled_trajectory))

    # Write a video of the trajectory
    vid = VideoMaker(video_filename)
    for z in sampled_trajectory:
        z = to_torch(z)
        hallucination = netG(z)
        preds = netC(z)
        pred_conf, pred_idx = torch.exp(preds).max(1)
        pred_conf, pred_idx = to_np(pred_conf)[0], to_np(pred_idx)[0]
        caption = "Class: {} (confidence {:.3f})".format(dataloader.lab_conv.labels[pred_idx], pred_conf)
        vid.write_frame(hallucination, caption)
    vid.finish()


def generate_comparison(networks, dataloader, **options):
    netG = networks['generator']
    netE = networks['encoder']
    netC = networks['classifier']
    netD = networks['discriminator']
    result_dir = options['result_dir']
    image_size = options['image_size']
    latent_size = options['latent_size']
    output_frame_count = options['counterfactual_frame_count']
    speed = options['speed']
    momentum_mu = options['momentum_mu']
    max_iters = options['counterfactual_max_iters']
    result_dir = options['result_dir']


    # Generate an NxN square visualization where N is the class size
    counterfactual_latent_points = []
    for _ in range(dataloader.num_classes):
        # Start with a random example
        most_likely_class, least_likely_class, start_score, start_img = select_uncertain_example(dataloader, netE, netC, pool_size=1)
        print("Got an example, the classifier thinks it is class: {}".format(most_likely_class))
        start_class = most_likely_class

        for class_idx in range(dataloader.num_classes):
            # Generate a path in latent space from start_img to a known classification
            z = netE(Variable(start_img))
            target_class = class_idx
            z_trajectory = generate_z_trajectory(z, target_class, netC, netE, netG, dataloader, speed, momentum_mu, max_iters=max_iters)
            print("Generating what the image would look like if it were class {}".format(target_class))
            counterfactual_latent_points.append(z_trajectory[-1])

    images = []
    original_img = start_img.cpu().numpy().squeeze(0)
    reencoded = to_np(netG(netE(Variable(start_img)))).squeeze(0)
    #images.append(original_img)
    #images.append(reencoded)
    for z in counterfactual_latent_points:
        img = to_np(netG(to_torch(z))).squeeze(0)
        images.append(img)

    images = np.array(images).transpose((0,2,3,1))
    video_filename = make_video_filename(result_dir, dataloader, start_class, start_class, label_type='grid')

    # Save the images in npy format to re-load as training data
    trajectory_filename = video_filename.replace('.mjpeg', '.npy')
    np.save(trajectory_filename, images)

    # Save the images in jpg format to display to the user
    imutil.show(images, filename=video_filename.replace('.mjpeg', '.jpg'))


def select_uncertain_example(dataloader, netE, netC, pool_size=100, reverse=False):
    print("Performing uncertainty sampling with dataloader {}".format(dataloader))
    images = []
    certainties = []
    most_likely_classes = []
    least_likely_classes = []

    for i, (img, label, attr) in enumerate(dataloader):
        images.append(img)
        z = netE(Variable(img))
        preds = torch.exp(netC(z))
        certainty, class_idx = preds.max(1)
        uncertainty, min_idx = preds.min(1)
        certainties.append(to_np(certainty)[0])
        most_likely_classes.append(to_np(class_idx)[0])
        least_likely_classes.append(to_np(min_idx)[0])
        if i > pool_size:
            break
    if reverse:
        idx = np.argmax(certainties)
    else:
        idx = np.argmin(certainties)
    selected_image = images[idx]
    most_likely_class = most_likely_classes[idx]
    least_likely_class = least_likely_classes[idx]
    selected_certainty = certainties[idx]
    return most_likely_class, least_likely_class, selected_certainty, selected_image


# Selects a class different from the start class
def random_target_class(dataloader, start_class):
    target_class = random.randint(0, dataloader.num_classes - 2)
    if start_class <= target_class:
        target_class += 1
    return target_class


def generate_z_trajectory(z, target_class, netC, netE, netG, dataloader,
        speed=.01, momentum_mu=.96, max_iters=100, spherical=True):
    # Generate z_trajectory
    z_trajectory = []
    z_trajectory.append(to_np(z))  # initial point
    momentum = Variable(torch.zeros(z.size())).cuda()
    target_label = torch.LongTensor(1)
    target_label[:] = int(target_class)
    target_label = Variable(target_label).cuda()
    original_z = z.clone()
    for i in range(max_iters):
        preds = netC(netE(netG(z)))
        #preds = netC(z)
        # NOTE: switch likelihood vs NLL here
        #cf_loss = nll_loss(preds, target_label)
        cf_loss = nll_loss(torch.log(preds), target_label)

        # Distance in latent space from original point
        cf_loss += .0001 * torch.sum((z - original_z) ** 2)

        dc_dz = autograd.grad(cf_loss, z, cf_loss, retain_graph=True)[0]
        momentum -= dc_dz * speed
        z += momentum
        momentum *= momentum_mu
        if spherical:
            l2_norm = torch.mul(z, z).sum()
            z /= l2_norm

        preds = netC(z)
        predicted_class = to_np(preds.max(1)[1])[0]
        # NOTE: Using softmax for now- can switch for numerical stability
        #pred_confidence = np.exp(to_np(preds.max(1)[0])[0])
        pred_confidence = to_np(preds.max(1)[0])[0]
        z_trajectory.append(to_np(z))
        predicted_class_name = dataloader.lab_conv.labels[predicted_class]
        print("Class: {} ({:.3f} confidence). Target class {}".format(
            predicted_class_name, pred_confidence, target_class))
        if pred_confidence > .90 and predicted_class == target_class:
            break
    return z_trajectory


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


def sample_trajectory(zt, output_sample_count):
    distances = np.array([np.linalg.norm(zt[i+1] - zt[i]) for i in range(len(zt) - 1)])
    total_distance = sum(distances)
    if total_distance == 0:
        print("Warning: Trajectory has zero length")
        return [zt[0]] * output_sample_count
    samples = []
    for i in range(output_sample_count):
        sample_distance = i * (total_distance / output_sample_count)
        samples.append(get_sample_point(zt, sample_distance, distances))
    return samples
        

def get_sample_point(zt, sample_distance, distances):
    for i in range(len(zt)):
        theta = (sample_distance - sum(distances[:i])) / distances[i]
        if theta <= 1:
            return (1 - theta) * zt[i] + theta * zt[i+1]
    assert False
