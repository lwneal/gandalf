from torch import nn


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class generatorReLU64(nn.Module):
    def __init__(self, latent_size=100):
        super(self.__class__, self).__init__()
        self.conv1 = nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(   512,    256, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(   256,    128, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(   128,     64, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(    64,      3, 4, 2, 1, bias=False)
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
        x = self.conv5(x)
        x = nn.Sigmoid()(x)
        return x


class discriminatorLReLU64(nn.Module):
    def __init__(self, latent_size=100):
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
        return x.view(-1, 1).squeeze(1)


class encoderLReLU64(nn.Module):
    def __init__(self, latent_size=100):
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
        x = self.fc1(x.squeeze())
        return x


class classifierMLP256(nn.Module):
    def __init__(self, num_classes, latent_size=100):
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
        x = nn.Softmax()(x)
        from torch.nn.functional import log_softmax
        x = log_softmax(x)
        return x
