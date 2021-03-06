import torch.nn as nn
from GAN_training.models.spectral_normalization import SpectralNorm
import numpy as np
import GAN_training.utils as utils

import pdb

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.conv_sc = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv_sc.weight.data, 1.)
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Sequential(
                nn.Upsample(scale_factor=2),
                self.conv_sc
                )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        self.stride = stride

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
                

    def forward(self, x):
        if self.stride == 1:
            return self.model(x)
        else:
            return self.model(x) + self.bypass(x)


# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.z_dim = opt.GAN_nz
        self.ngpu = opt.ngpu
        ngf = self.ngf = 512
        self.dense = nn.Linear(self.z_dim, 6 * 6 * ngf)
        self.final = nn.Conv2d(ngf // 8, opt.nc, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.network = nn.Sequential(
            ResBlockGenerator(ngf, ngf // 2, stride=2),
            ResBlockGenerator(ngf // 2, ngf // 4, stride=2),
            ResBlockGenerator(ngf // 4, ngf // 8, stride=2),
            nn.BatchNorm2d(ngf // 8),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, input_noise):
        layer1_out = self.dense(input_noise)
        layer1_out = layer1_out.view(-1, self.ngf, 6, 6)
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.network, layer1_out, range(self.ngpu))
        else:
            output = self.network(layer1_out)
        return output


class Discriminator(nn.Module):
    """
    Miyato 2018
    """
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        ndf = 64
        self.ngpu = int(opt.ngpu)
        self.feat_net = nn.Sequential(
            FirstResBlockDiscriminator(opt.nc, ndf, stride=2),
            ResBlockDiscriminator(ndf, ndf * 2, stride=2),
            ResBlockDiscriminator(ndf * 2, ndf * 4, stride=2),
            ResBlockDiscriminator(ndf * 4, ndf * 8, stride=2),
            ResBlockDiscriminator(ndf * 8, ndf * 16, stride=1),
            nn.ReLU(),
        )

        self.disc_net = nn.Sequential(
            SpectralNorm(nn.Conv2d(ndf * 16, 1, 3, 1, 0, bias=False))
        )
        self.aux_net = nn.Sequential(
            SpectralNorm(nn.Conv2d(ndf * 16, opt.num_classes, 3, 1, 0, bias=False))
        )
        self.disc_net.apply(utils.weights_init_spectral)
        self.softmax = nn.Softmax()

    def forward(self, input):
        if self.ngpu > 1:
            feat = nn.parallel.data_parallel(self.feat_net, input, range(self.ngpu))
            disc_logits = nn.parallel.data_parallel(self.disc_net, feat, range(self.ngpu))
            aux_logits = nn.parallel.data_parallel(self.aux_net, feat, range(self.ngpu))
        else:
            feat = self.feat_net(input)
            
            disc_logits = self.disc_net(feat)
            aux_logits = self.aux_net(feat)

        return disc_logits.view(-1, 1).squeeze(1), self.softmax(aux_logits).squeeze()

class ClassifierMultiHinge(nn.Module):
    def __init__(self, opt):
        super(ClassifierMultiHinge, self).__init__()

        self.opt = opt
        ndf = 64
        self.ngpu = int(opt.ngpu)
        self.feat_net = nn.Sequential(
            FirstResBlockDiscriminator(opt.nc, ndf, stride=2),
            ResBlockDiscriminator(ndf, ndf * 2, stride=2),
            ResBlockDiscriminator(ndf * 2, ndf * 4, stride=2),
            ResBlockDiscriminator(ndf * 4, ndf * 8, stride=2),
            ResBlockDiscriminator(ndf * 8, ndf * 16, stride=1),
            nn.ReLU(),
        )

        self.aux_net = nn.Sequential(
            SpectralNorm(nn.Conv2d(ndf * 16, opt.num_classes, 3, 1, 0, bias=False))
        )
        self.aux_net.apply(utils.weights_init_spectral)

    def forward(self, input):
        if self.ngpu > 1:
            feat = nn.parallel.data_parallel(self.feat_net, input, range(self.ngpu))
            aux_logits = nn.parallel.data_parallel(self.aux_net, feat, range(self.ngpu))
        else:
            feat = self.feat_net(input)
            aux_logits = self.aux_net(feat)

        output = aux_logits.squeeze()
        return output[:,self.opt.num_classes-1], output


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.opt = opt
        ndf = 128
        self.ngpu = int(opt.ngpu)
        self.feat_net = nn.Sequential(
            FirstResBlockDiscriminator(opt.nc, ndf, stride=2),
            ResBlockDiscriminator(ndf, ndf, stride=2),
            ResBlockDiscriminator(ndf, ndf, stride=2),
            ResBlockDiscriminator(ndf, ndf, stride=2),
            nn.ReLU(),
        )

        self.enc_net = nn.Sequential(
            SpectralNorm(nn.Conv2d(ndf, self.opt.GAN_nz, 2, 1, 0, bias=False))
        )
        self.enc_net.apply(utils.weights_init_spectral)

    def forward(self, input):
        if self.ngpu > 1:
            feat = nn.parallel.data_parallel(self.feat_net, input, range(self.ngpu))
            code = nn.parallel.data_parallel(self.enc_net, feat, range(self.ngpu))
        else:
            feat = self.feat_net(input)
            code = self.enc_net(feat)

        return code.view(code.size(0), self.opt.GAN_nz)
