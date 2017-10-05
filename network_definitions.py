import torch
from torch import nn
from torch.nn.functional import log_softmax


def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class generator28(nn.Module):
    def __init__(self, latent_size=10, **kwargs):
        super(self.__class__, self).__init__()
        Z = latent_size
        # in_channels, out_channels, kernel_size, stride, padding
        self.conv1 = nn.ConvTranspose2d(     Z,    256, 7, 1, 0, bias=False)
        self.conv2 = nn.ConvTranspose2d(   256,    128, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(   128,      3, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, True)(x)
        x = self.conv3(x)
        x = nn.Sigmoid()(x)
        return x


class discriminator28(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        # 3x28x28
        self.conv1 = nn.Conv2d(3,       64,      4, 2, 1, bias=False)
        # 64x14x14
        self.conv2 = nn.Conv2d(64,      128,     4, 2, 1, bias=False)
        # 128x7x7
        self.conv3 = nn.Conv2d(128,     256,     4, 2, 1, bias=False)
        # 256x3x3
        self.conv4 = nn.Conv2d(256,     1,     3, 1, 0, bias=False)
        # 1x1x1
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv4(x)
        # Global average pooling
        x = x.mean(-1).mean(-1)
        return x.view(-1, 1).squeeze(1)


class encoder28(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        # 3x28x28
        self.conv1 = nn.Conv2d(3,       64,      4, 2, 1, bias=False)
        # 64x14x14
        self.conv2 = nn.Conv2d(64,      128,     4, 2, 1, bias=False)
        # 128x7x7
        self.conv3 = nn.Conv2d(128,     256,     4, 2, 1, bias=False)
        # 256x3x3
        self.conv4 = nn.Conv2d(256,     latent_size,     3, 1, 0, bias=False)
        # 1x1x1
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv4(x)
        return x.squeeze(-1).squeeze(-1)


class encoder28dropout(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        # 3x28x28
        self.conv1 = nn.Conv2d(3,       64,      4, 2, 1, bias=False)
        # 64x14x14
        self.conv2 = nn.Conv2d(64,      128,     4, 2, 1, bias=False)
        # 128x7x7
        self.conv3 = nn.Conv2d(128,     256,     4, 2, 1, bias=False)
        # 256x3x3
        self.conv4 = nn.Conv2d(256,     latent_size,     3, 1, 0, bias=False)
        self.conv4_drop = nn.Dropout2d()
        # 1x1x1
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv4(x)
        x = self.conv4_drop(x)
        return x.squeeze(-1).squeeze(-1)


class generator32(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.conv1 = nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False)
        self.conv2 = nn.ConvTranspose2d(   512,    256, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(   256,    128, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(   128,     3, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv4(x)
        x = nn.Sigmoid()(x)
        return x


class generatorReLU64(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.conv1 = nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False)
        self.conv2 = nn.ConvTranspose2d(   512,    256, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(   256,    128, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(   128,     64, 4, 2, 1, bias=False)
        self.conv5 = nn.ConvTranspose2d(    64,      3, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(True)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU(True)(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.ReLU(True)(x)
        x = self.conv5(x)
        x = nn.Sigmoid()(x)
        return x


class generatorReLU128(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        Z = latent_size
        self.conv1 = nn.ConvTranspose2d(     Z,   1024, 4, 1, 0, bias=False)
        self.conv2 = nn.ConvTranspose2d(  1024,    512, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(   512,    256, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(   256,    128, 4, 2, 1, bias=False)
        self.conv5 = nn.ConvTranspose2d(   128,     64, 4, 2, 1, bias=False)
        self.conv6 = nn.ConvTranspose2d(    64,      3, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(True)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU(True)(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.ReLU(True)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.ReLU(True)(x)
        x = self.conv6(x)
        x = nn.Sigmoid()(x)
        return x

#LeakyReLU
class generatorLeakyReLU128(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        Z = latent_size
        self.conv1 = nn.ConvTranspose2d(     Z,   1024, 4, 1, 0, bias=False)
        self.conv2 = nn.ConvTranspose2d(  1024,    512, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(   512,    256, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(   256,    128, 4, 2, 1, bias=False)
        self.conv5 = nn.ConvTranspose2d(   128,     64, 4, 2, 1, bias=False)
        self.conv6 = nn.ConvTranspose2d(    64,      3, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(True)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(True)(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(True)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(True)(x)
        x = self.conv6(x)
        x = nn.Sigmoid()(x)
        return x


class discriminator32(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(3,      128,     4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(128,     256,     4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(256,     512,     4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(512,     1,       4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv4(x)
        # Global average pooling
        x = x.mean(-1).mean(-1)
        return x.view(-1, 1).squeeze(1)


class discriminatorLReLU64(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(3,       64,      4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      128,     4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128,     256,     4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256,     512,     4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512,     1,       4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv5(x)
        # Global average pooling
        x = x.mean(-1).mean(-1)
        return x.view(-1, 1).squeeze(1)


class discriminatorMultiscale128With4Critics(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(3,       64,      4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      128,     4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128,     256,     4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256,     512,     4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512,     1024,    4, 2, 1, bias=False)
        self.conv_c1 = nn.Conv2d(128,     1,       4, 1, 0, bias=False)
        self.conv_c2 = nn.Conv2d(256,     1,       4, 1, 0, bias=False)
        self.conv_c3 = nn.Conv2d(512,     1,       4, 1, 0, bias=False)
        self.conv_c4 = nn.Conv2d(1024,    1,       4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        critic1 = self.conv_c1(x).mean(-1).mean(-1)

        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        critic2 = self.conv_c2(x).mean(-1).mean(-1)

        x = self.conv4(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        critic3 = self.conv_c3(x).mean(-1).mean(-1)

        x = self.conv5(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        critic4 = self.conv_c4(x).mean(-1).mean(-1)

        wasserstein_distance = sum([critic1, critic2, critic3, critic4])
        return wasserstein_distance.squeeze()


class discriminatorMultiscale128(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(3,       64,      4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      128,     4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128,     256,     4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256,     512,     4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512,     1024,    4, 2, 1, bias=False)
        self.conv6 = nn.Conv2d(1024,    1,       4, 1, 0, bias=False)
        self.conv_c1 = nn.Conv2d(256,     1,       4, 1, 0, bias=False)
        self.conv_c2 = nn.Conv2d(512,     1,       4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        # This skin doesn't look real...
        critic1 = self.conv_c1(x).mean(-1).mean(-1)

        x = self.conv4(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        # This nose doesn't look real...
        critic2 = self.conv_c2(x).mean(-1).mean(-1)

        x = self.conv5(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv6(x)

        # This face doesn't look real!
        critic3 = x.mean(-1).mean(-1)

        wasserstein_distance = critic1 + critic2 + critic3
        return wasserstein_distance.squeeze()


class encoderLReLU64(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(3,       64,     4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      128,    4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128,     256,    4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256,     512,    4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512,     1024,   4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(1024, self.latent_size, bias=False)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv5(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = x.squeeze()
        x = self.fc1(x)
        return x


class encoder32(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(3,      128,    4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(128,     256,    4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(256,     512,    4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(512,     1024,   4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(1024, self.latent_size, bias=False)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv4(x)
        x = x.squeeze()
        x = self.fc1(x)
        return x


class encoder32sphere(nn.Module):
    def __init__(self, latent_size=100, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(3,      128,    4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(128,     256,    4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(256,     512,    4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(512,     1024,   4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(1024, self.latent_size, bias=False)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv4(x)
        x = x.squeeze()
        x = self.fc1(x)
        xnorm = torch.norm(x, p=2, dim=1).detach()
        xnorm = xnorm.expand(1, x.size()[0])
        xnorm = xnorm.transpose(1,0)
        x = x.div(xnorm)
        return x


class encoderLReLU128(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        # in_channels, out_channels, kernel_size, stride, padding, dilation
        self.conv1 = nn.Conv2d(3,       64,     4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      128,    4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128,     256,    4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256,     512,    4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512,     1024,   4, 2, 1, bias=False)
        self.conv6 = nn.Conv2d(1024,    2048,   4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(2048, self.latent_size, bias=False)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv6(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = x.squeeze()
        x = self.fc1(x)
        return x


class classifierMLP256(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.fc2(x)
        x = log_softmax(x)
        return x
