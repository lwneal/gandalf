import os
import network_definitions
import torch
from torch import optim


def build_networks(latent_size,
        result_dir,
        image_size,
        num_classes,
        encoder_network_name,
        generator_network_name,
        discrim_network_name):

    networks = {}

    EncoderClass = getattr(network_definitions, encoder_network_name)
    networks['encoder'] = EncoderClass(latent_size)

    GeneratorClass = getattr(network_definitions, generator_network_name)
    networks['generator'] = GeneratorClass(latent_size)

    DiscrimClass = getattr(network_definitions, discrim_network_name)
    networks['discriminator'] = DiscrimClass(latent_size)

    ClassifierClass = network_definitions.classifierMLP256
    networks['classifier'] = ClassifierClass(latent_size, num_classes=num_classes)

    for name, net in networks.items():
        pth = get_latest_pth(result_dir, name)
        if pth:
            print("Loading {} from checkpoint {}".format(name, pth))
            net.load_state_dict(torch.load(pth))
    return networks


def get_optimizers(networks, lr, beta1):
    optimizers = {}
    for name in networks:
        net = networks[name]
        optimizers[name] = optim.Adam(net.parameters(), lr=lr, betas=(beta1, .999))
    return optimizers


def get_latest_pth(result_dir, name):
    files = os.listdir(result_dir)
    files = [f for f in files if f.startswith(name) and f.endswith('.pth')]
    if not files:
        return None
    files = [os.path.join(result_dir, f) for f in files]
    ordered_by_mtime = sorted(files, key=lambda x: os.stat(x).st_mtime)
    return ordered_by_mtime[-1]
    
