import torch
from torch.autograd import Variable
import os
import numpy as np
from imutil import show


def run_example_code(dataloader, nets, **options):
    netE = nets['netE']
    netG = nets['netG']

    print("My favorite number is {}".format(options['example_parameter']))
    
    print("I have {} examples in the {} set".format(
        len(dataloader), options['fold']))

    print("This is what a random image looks like:")
    image_size = options['image_size']
    image = np.random.rand((3, image_size, image_size))
    show(image)

    print("Encoding the image down to {} dimensions...".format(options['latent_size']))
    image_tensor = torch.autograd.Variable(torch.FloatTensor(np.expand_dims(image, 0)))
    z = netE(image_tensor)
    z = z.data.cpu().numpy()
    print(z)

    print("Decoding the image back to the manifold")
    z = torch.autograd.Variable(torch.FloatTensor(z))
    reconstructed = netG(z)
    reconstructed = reconstructed.data.cpu().numpy()

    print("This is what the image looks like after it's been autoencoded:")
    show(reconstructed)

    r_error = np.mean((image - reconstructed) ** 2)

    my_results = {
            'reconstruction_error': r_error,
            'dataset_size': len(dataloader),
    }
    return my_results
