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


class generator32(nn.Module):
    def __init__(self, latent_size=100, **kwargs):
        super(self.__class__, self).__init__()
        self.conv1 = nn.ConvTranspose2d(latent_size,     512, 4, 1, 0, bias=False)
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


class multiclassDiscriminator32(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,      128,     4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(128,     256,     4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(256,     512,     4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(512,     512,       4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512,     1024,       2, 1, 0, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(1024, num_classes)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)
        x = self.conv1(x)
        x = self.bn0(x)
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
        x = x.view(batch_size, -1)
        if return_features:
            return x
        x = self.fc1(x)
        return x
