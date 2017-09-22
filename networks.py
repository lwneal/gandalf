import os
import network_definitions
import torch
from torch import optim


def build_networks(latent_size, result_dir, image_size, num_classes):

    networks = {}
    if image_size <= 64:
        networks['encoder'] = network_definitions.encoderLReLU64(latent_size)
        networks['generator'] = network_definitions.generatorReLU64(latent_size)
    else:
        networks['encoder'] = network_definitions.encoderLReLU128(latent_size)
        networks['generator'] = network_definitions.generatorReLU128(latent_size)

    networks['discriminator'] = network_definitions.discriminatorMultiscale128()
    networks['classifier'] = network_definitions.classifierMLP256(num_classes=num_classes, latent_size=latent_size)

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
    
