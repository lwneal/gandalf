import torch
from torch import nn
from torch.nn.functional import log_softmax


def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class generator32(nn.Module):
    def __init__(self, latent_size=100, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.fc1 = nn.Linear(latent_size, 4*4*512)
        self.conv1 = nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=False)
        self.conv2 = nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=False)
        self.conv3 = nn.ConvTranspose2d(   128,        3, 4, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm1d(4*4*512)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)

        self.batch_size = batch_size
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        """
        Based on Improved GAN from Salimans et al
        For reference:

        nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu), g=None)
        ll.ReshapeLayer(gen_layers[-1], (args.batch_size,512,4,4))
        nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None) # 4 -> 8
        nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None) # 8 -> 16
        nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1) # 16 -> 32
        """
        batch_size = x.shape[0]
        x = self.fc1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.bn0(x)
        x = x.resize(batch_size, 512, 4, 4)
        # 512 x 4 x 4
        x = self.conv1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.bn1(x)
        # 256 x 8 x 8
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.bn2(x)
        # 128 x 16 x 16
        x = self.conv3(x)
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
        self.fc1 = nn.Linear(1024, num_classes)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        """
        Based on Salimans et al

        Reference:
        ll.DropoutLayer(disc_layers[-1], p=0.2)
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 64, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu))
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 64, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu))
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 64, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu))
        ll.DropoutLayer(disc_layers[-1], p=0.5)
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu))
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu))
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu))
        ll.DropoutLayer(disc_layers[-1], p=0.5)
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu))
        nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=128, W=Normal(0.05), nonlinearity=nn.lrelu))
        nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=128, W=Normal(0.05), nonlinearity=nn.lrelu))
        ll.GlobalPoolLayer(disc_layers[-1])
        nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=10, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1)
        """
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
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = x.view(batch_size, -1)
        if return_features:
            return x
        x = self.fc1(x)
        return x
