import os
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_spectral(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight_u.data.normal_(0.0, 0.02)
        m.weight_v.data.normal_(0.0, 0.02)
        m.weight_bar.data.normal_(0.0, 0.02)

def sample_normal(bs, nz):
    return torch.FloatTensor(bs, nz).normal_(0, 1)

def mkdirp(d):
    if not os.path.isdir(d):
        os.makedirs(d)
