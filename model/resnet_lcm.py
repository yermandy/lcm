import torch
import torch.nn as nn

from model.lcm import LCM, MultiBiasedLinear
from model.resnet import ResNet, Bottleneck


class ResNetLCM(ResNet):
    def __init__(self, datasets_len, layers, **kwargs):
        super(ResNetLCM, self).__init__(Bottleneck, layers, **kwargs)
        self.dataset_ages = 180
        self.mbl = MultiBiasedLinear(self.fc.in_features, self.dataset_ages, datasets_len)
        self.fc = nn.Sequential()
        self.lcm = LCM(datasets_len)

    def forward(self, x, d, return_PagIx=False):
        а = super().forward(x)
        x = self.mbl(x, d)
        x = self.lcm(x, d, return_PagIx=return_PagIx)
        return x


def resnet18lcm(datasets_len, **kwargs):
    return ResNetLCM(datasets_len, [2, 2, 2, 2], model_name='resnet18lcm', **kwargs)


def resnet50lcm(datasets_len, **kwargs):
    return ResNetLCM(datasets_len, [3, 4, 6, 3], model_name='resnet50lcm', **kwargs)