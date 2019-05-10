'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

import pdb


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x, feat=False):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        if feat:
            
            relu1_2 = self.features[5](x)
            relu1_2 = relu1_2.view(relu1_2.size(0),-1)
            relu2_2 = self.features[12](x)
            relu2_2 = relu2_2.view(relu2_2.size(0),-1)
            relu3_3 = self.features[25](x)
            relu3_3 = relu3_3.view(relu3_3.size(0),-1)
            relu4_3 = self.features[38](x)
            relu4_3 = relu4_3.view(relu4_3.size(0),-1)
            
            return torch.cat((relu1_2, relu2_2, relu3_3, relu4_3),1)
        else:
            return self.classifier(out)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
