import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LCM, MultiBiasedLinear
from resnet import ResNet, Bottleneck


class ResNet50LCM(ResNet):
    def __init__(self, datasets_len, **kwargs):
        super(ResNet50LCM, self).__init__(Bottleneck, [3, 4, 6, 3], **kwargs)
        self.model_name = 'lcm'
        self.mbl = MultiBiasedLinear(self.fc.in_features, 180, datasets_len)
        self.fc = nn.Sequential()
        self.lcm = LCM(datasets_len)

    def forward(self, x, d, return_PagIx=False):
        x = super().forward(x)
        x = self.mbl(x, d)
        x = self.lcm(x, d, return_PagIx=return_PagIx)
        return x


def resnet50_lcm(datasets_len, **kwargs):
    return ResNet50LCM(datasets_len, **kwargs)