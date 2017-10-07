import torch
from torch.autograd import Variable
import os
import numpy as np
from imutil import show


def run_example_code(nets, dataloader, **options):
    netE = nets['encoder']
    netG = nets['generator']

    print("My favorite number is {}".format(options['example_parameter']))
    
    print("I have {} examples in the {} set".format(
        len(dataloader), options['fold']))

    print("This is what a random image looks like:")
    image_size = options['image_size']
    # Torch uses <channels, height, width>
    image = np.random.rand(3, image_size, image_size)
    x = np.expand_dims(image, 0)
    x = torch.autograd.Variable(torch.FloatTensor(x))
    x = x.cuda()
    show(x)

    print("Encoding the image down to {} dimensions...".format(options['latent_size']))
    z = netE(x)
    z = torch_to_np(z)

    print("Latent vector is: {}".format(z))

    print("Decoding the latent vector back to an image...")
    z = np_to_torch(z)
    outputs = netG(z)
    reconstructed = torch_to_np(outputs[0])

    print("This is what the image looks like after it's been autoencoded:")
    # Shuffle it back to <height, width, channels>
    show(reconstructed.transpose((1,2,0)))

    r_error = np.mean((image - reconstructed) ** 2)

    my_results = {
            'reconstruction_error': r_error,
            'dataset_size': len(dataloader),
    }
    return my_results


def np_to_torch(x):
    return torch.autograd.Variable(torch.FloatTensor(x)).cuda()


def torch_to_np(x):
    return x.data.cpu().numpy()
