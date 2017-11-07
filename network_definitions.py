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


class discriminator28instancenorm(nn.Module):
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
        self.norm1 = nn.InstanceNorm2d(128)
        self.norm2 = nn.InstanceNorm2d(256)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv2(x)
        x = self.norm1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv3(x)
        x = self.norm2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv4(x)
        # Global average pooling
        x = x.mean(-1).mean(-1)
        return x.view(-1, 1).squeeze(1)


class discriminator28dropout(nn.Module):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__()
        # 3x28x28
        self.conv1 = nn.Conv2d(3,       128,      4, 2, 1, bias=False)
        # 64x14x14
        self.conv2 = nn.Conv2d(128,      256,     4, 2, 1, bias=False)
        # 128x7x7
        self.conv3 = nn.Conv2d(256,     512,     4, 2, 1, bias=False)
        # 256x3x3
        self.conv4 = nn.Conv2d(512,     1,     3, 1, 0, bias=False)
        # 1x1x1
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout1 = nn.Dropout2d(p=0.4)
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
        x = self.dropout1(x)

        # Global average pooling
        x = x.mean(-1).mean(-1)
        return x.view(-1, 1).squeeze(1)


class discriminator28dropoutlog(nn.Module):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__()
        # 3x28x28
        self.conv1 = nn.Conv2d(3,       128,      4, 2, 1, bias=False)
        # 64x14x14
        self.conv2 = nn.Conv2d(128,      256,     4, 2, 1, bias=False)
        # 128x7x7
        self.conv3 = nn.Conv2d(256,     512,     4, 2, 1, bias=False)
        # 256x3x3
        self.conv4 = nn.Conv2d(512,     1,     3, 1, 0, bias=False)
        # 1x1x1
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout1 = nn.Dropout2d(p=0.5)
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
        x = self.dropout1(x)

        # Global average pooling
        x = x.mean(-1).mean(-1)
        x = x.view(-1, 1).squeeze(1)

        # Limit magnitude with a heuristic log function
        x = torch.sign(x) * torch.log(torch.abs(x) + 1)
        return x



class discriminator28nobn(nn.Module):
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
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv3(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.conv4(x)
        # Global average pooling
        x = x.mean(-1).mean(-1)
        return x.view(-1, 1).squeeze(1)


class discriminator28tanh(nn.Module):
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
        x = x.view(-1, 1).squeeze(1)

        x = nn.Tanh()(x)
        return x

class discriminator28log(nn.Module):
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
        x = x.view(-1, 1).squeeze(1)

        # Limit magnitude with a heuristic log function
        x = torch.sign(x) * torch.log(torch.abs(x) + 1)
        return x


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


class generator40(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.conv1 = nn.ConvTranspose2d(latent_size, 512, 5, 1, 0, bias=False)
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
        x = nn.ReLU(True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(True)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU(True)(x)
        x = self.conv4(x)
        x = nn.Sigmoid()(x)
        return x


class generator64(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        # A branch for the low-rank parts of images we can encode
        self.conv1 = nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False)
        self.conv2 = nn.ConvTranspose2d(   512,    256, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(   256,    128, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(   128,     64, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)

        # Another branch for the exogenous stuff we can't encode
        self.rconv1 = nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False)
        self.rconv2 = nn.ConvTranspose2d(   512,    256, 4, 2, 1, bias=False)
        self.rconv3 = nn.ConvTranspose2d(   256,    128, 4, 2, 1, bias=False)
        self.rconv4 = nn.ConvTranspose2d(   128,     64, 4, 2, 1, bias=False)
        self.rbn1 = nn.BatchNorm2d(512)
        self.rbn2 = nn.BatchNorm2d(256)
        self.rbn3 = nn.BatchNorm2d(128)
        self.rbn4 = nn.BatchNorm2d(64)

        # They come together at the very end
        self.conv5 = nn.ConvTranspose2d(    128,     3, 4, 2, 1, bias=False)
        self.apply(weights_init)
        self.cuda()

    def forward(self, latent_vector, exo_noise):
        x = latent_vector.unsqueeze(-1).unsqueeze(-1)
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
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        # r is for residual
        r = exo_noise.unsqueeze(-1).unsqueeze(-1)
        r = self.rconv1(r)
        r = self.rbn1(r)
        r = nn.LeakyReLU(0.2, inplace=True)(r)
        r = self.rconv2(r)
        r = self.rbn2(r)
        r = nn.LeakyReLU(0.2, inplace=True)(r)
        r = self.rconv3(r)
        r = self.rbn3(r)
        r = nn.LeakyReLU(0.2, inplace=True)(r)
        r = self.rconv4(r)
        r = self.rbn4(r)
        r = nn.LeakyReLU(0.2, inplace=True)(r)

        xr = torch.cat([x, r], dim=1)
        x = self.conv5(xr)
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


class generator128exo(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        Z = latent_size
        self.conv1 = nn.ConvTranspose2d(     Z,   1024, 4, 1, 0, bias=False)
        self.conv2 = nn.ConvTranspose2d(  1024,    512, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(   512,    256, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(   256,    128, 4, 2, 1, bias=False)
        self.conv5 = nn.ConvTranspose2d(   128,     64, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)

        self.rconv1 = nn.ConvTranspose2d(     Z,   1024, 4, 1, 0, bias=False)
        self.rconv2 = nn.ConvTranspose2d(  1024,    512, 4, 2, 1, bias=False)
        self.rconv3 = nn.ConvTranspose2d(   512,    256, 4, 2, 1, bias=False)
        self.rconv4 = nn.ConvTranspose2d(   256,    128, 4, 2, 1, bias=False)
        self.rconv5 = nn.ConvTranspose2d(   128,     64, 4, 2, 1, bias=False)
        self.rbn1 = nn.BatchNorm2d(1024)
        self.rbn2 = nn.BatchNorm2d(512)
        self.rbn3 = nn.BatchNorm2d(256)
        self.rbn4 = nn.BatchNorm2d(128)
        self.rbn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.ConvTranspose2d(   128,      3, 4, 2, 1, bias=False)

        self.apply(weights_init)
        self.cuda()

    def forward(self, z, exo_noise):
        x = z.unsqueeze(-1).unsqueeze(-1)
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

        r = exo_noise.unsqueeze(-1).unsqueeze(-1)
        r = self.rconv1(r)
        r = self.rbn1(r)
        r = nn.LeakyReLU(True)(r)
        r = self.rconv2(r)
        r = self.rbn2(r)
        r = nn.LeakyReLU(True)(r)
        r = self.rconv3(r)
        r = self.rbn3(r)
        r = nn.LeakyReLU(True)(r)
        r = self.rconv4(r)
        r = self.rbn4(r)
        r = nn.LeakyReLU(True)(r)
        r = self.rconv5(r)
        r = self.rbn5(r)
        r = nn.LeakyReLU(True)(r)

        xr = torch.cat([x, r], dim=1)
        x = self.conv6(xr)
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


class discriminator32instancenorm(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(3,      128,     4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(128,     256,     4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(256,     512,     4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(512,     1,       4, 1, 0, bias=False)
        self.ln1 = nn.InstanceNorm2d(256)
        self.ln2 = nn.InstanceNorm2d(512)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.ln1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.ln2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv4(x)
        # Global average pooling
        x = x.mean(-1).mean(-1)
        return x.view(-1, 1).squeeze(1)


class discriminator32log(nn.Module):
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
        x = x.view(-1, 1).squeeze(1)
        # Limit magnitude with a heuristic log function
        x = torch.sign(x) * torch.log(torch.abs(x) + 1)
        return x


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
discriminator64 = discriminatorLReLU64


class discriminator40(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(3,       64,      4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      128,     4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128,     256,     4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256,     512,     4, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(512,     1,       4, 1, 0, bias=False)
        self.bn1 = nn.InstanceNorm2d(128)
        self.bn2 = nn.InstanceNorm2d(256)
        self.bn3 = nn.InstanceNorm2d(512)
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


class discriminator64instancenorm(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(3,       64,      4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      128,     4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128,     256,     4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256,     512,     4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512,     1,       4, 1, 0, bias=False)
        self.bn1 = nn.InstanceNorm2d(128)
        self.bn2 = nn.InstanceNorm2d(256)
        self.bn3 = nn.InstanceNorm2d(512)
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


class discriminator64log(nn.Module):
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
        # Limit magnitude with a heuristic log function
        x = torch.sign(x) * torch.log(torch.abs(x) + 1)
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
discriminator128 = discriminatorMultiscale128


class discriminator128instancenorm(nn.Module):
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
        self.bn1 = nn.InstanceNorm2d(128)
        self.bn2 = nn.InstanceNorm2d(256)
        self.bn3 = nn.InstanceNorm2d(512)
        self.bn4 = nn.InstanceNorm2d(1024)
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

class encoder28LargeFilter(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        # 3x28x28
        self.conv1 = nn.Conv2d(3,       128,      4, 2, 1, bias=False)
        # 64x14x14
        self.conv2 = nn.Conv2d(128,      512,     4, 2, 1, bias=False)
        # 128x7x7
        self.conv3 = nn.Conv2d(512,     2048,     4, 2, 1, bias=False)
        # 256x3x3
        self.conv4 = nn.Conv2d(2048,     latent_size,     3, 1, 0, bias=False)
        # 1x1x1
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(2048)
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


class encoder28sphere(nn.Module):
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
        #self.fc1 = nn.Linear(1024, self.latent_size, bias=False)
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
        x = x.squeeze(-1).squeeze(-1)

        xnorm = torch.norm(x, p=2, dim=1).detach()
        xnorm = xnorm.expand(1, x.size()[0])
        xnorm = xnorm.transpose(1,0)
        x = x.div(xnorm)
        return x


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


class encoderLReLU64sphere(nn.Module):
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
        x = x.squeeze(-1).squeeze(-1)
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
        x = x.squeeze(-1).squeeze(-1)
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
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        xnorm = torch.norm(x, p=2, dim=1).detach()
        xnorm = xnorm.expand(1, x.size()[0])
        xnorm = xnorm.transpose(1,0)
        x = x.div(xnorm)
        return x


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
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        return x
encoder64 = encoderLReLU64


class encoder64sphere(nn.Module):
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
        x = x.squeeze(-1).squeeze(-1)

        x = self.fc1(x)
        xnorm = torch.norm(x, p=2, dim=1).detach()
        xnorm = xnorm.expand(1, x.size()[0])
        xnorm = xnorm.transpose(1,0)
        x = x.div(xnorm)
        return x


class encoder40(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(3,       64,     4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      128,    4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128,     256,    4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256,     512,    4, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(512,     1024,   4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(1024, self.latent_size, bias=False)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU(True)(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = nn.ReLU(True)(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = nn.ReLU(True)(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = nn.ReLU(True)(x)
        x = self.conv5(x)
        x = nn.ReLU(True)(x)
        x = x.squeeze(-1).squeeze(-1)

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
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        return x
encoder128 = encoderLReLU128



class encoder128sphere(nn.Module):
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
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        xnorm = torch.norm(x, p=2, dim=1).detach()
        xnorm = xnorm.expand(1, x.size()[0])
        xnorm = xnorm.transpose(1,0)
        x = x.div(xnorm)
        return x


class classifierMLP256(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 512)
        self.activ1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = log_softmax(x)
        return x


class classifierMulticlass(nn.Module):
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
        x = nn.Sigmoid()(x)
        return x
