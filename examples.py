import torch
from torch.autograd import Variable
import os
import numpy as np
from imutil import show


def run_example_code(nets, dataloader, **options):
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

    my_results = {
            'dataset_size': len(dataloader),
    }
    return my_results


def np_to_torch(x):
    return torch.autograd.Variable(torch.FloatTensor(x)).cuda()


def torch_to_np(x):
    return x.data.cpu().numpy()
