import network_definitions
from torch import optim


def build_networks(latentSize):
    netE = network_definitions.encoderLReLU64(latentSize)
    netG = network_definitions.generatorReLU64(latentSize)
    netD = network_definitions.discriminatorLReLU64()

    # TODO: Find the most recent saved model in resultsDir and load it
    """
    if opt.netE:
        netE.load_state_dict(torch.load(opt.netE))
    if opt.netG:
        netG.load_state_dict(torch.load(opt.netG))
    if opt.netD:
        netD.load_state_dict(torch.load(opt.netD))
    """
    return {
        'encoder': netE,
        'generator': netG,
        'discriminator': netD,
    }

def get_optimizers(networks, lr, beta1):
    optimizers = {}
    for name in networks:
        net = networks[name]
        optimizers[name] = optim.Adam(net.parameters(), lr=lr, betas=(beta1, .999))
    return optimizers

