import os
import network_definitions
import torch
from torch import optim
from torch import nn


def build_networks(num_classes, latent_size=10, image_size=32, **options):
    networks = {}

    EncoderClass = get_network_class(options['encoder'])
    networks['encoder'] = EncoderClass(latent_size)

    GeneratorClass = get_network_class(options['generator'])
    networks['generator'] = GeneratorClass(latent_size)

    DiscrimClass = get_network_class(options['discriminator'])
    networks['discriminator'] = DiscrimClass(latent_size)

    ClassifierClass = network_definitions.classifierMLP256
    networks['classifier'] = ClassifierClass(latent_size, num_classes=num_classes)

    for name, net in networks.items():
        pth = get_latest_pth(options['result_dir'], name)
        if pth:
            print("Loading {} from checkpoint {}".format(name, pth))
            net.load_state_dict(torch.load(pth))
    return networks


def get_network_class(name):
    if not hasattr(network_definitions, name):
        print("Error: could not construct network '{}'".format(name))
        print("Available networks are:")
        for net_name in dir(network_definitions):
            classobj = getattr(network_definitions, net_name)
            if type(classobj) is type and issubclass(classobj, nn.Module):
                print('\t' + net_name)
        raise AttributeError
    return getattr(network_definitions, name)


def save_networks(networks, epoch, result_dir):
    for name in networks:
        weights = networks[name].state_dict()
        filename = '{}/{}_epoch_{:02d}.pth'.format(result_dir, name, epoch)
        torch.save(weights, filename)


def get_optimizers(networks, lr=.001, beta1=.5, beta2=.999, **options):
    optimizers = {}
    for name in networks:
        net = networks[name]
        optimizers[name] = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))
    return optimizers


def get_latest_pth(result_dir, name):
    files = os.listdir(result_dir)
    files = [f for f in files if f.startswith(name) and f.endswith('.pth')]
    if not files:
        return None
    files = [os.path.join(result_dir, f) for f in files]
    ordered_by_mtime = sorted(files, key=lambda x: os.stat(x).st_mtime)
    return ordered_by_mtime[-1]
    
