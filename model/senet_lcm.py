import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lcm import LCM, MultiBiasedLinear
from model.resnet import ResNet
from model.senet import SEBasicBlock


class SENetLCM(ResNet):
    def __init__(self, datasets_len, layers, **kwargs):
        super(SENetLCM, self).__init__(SEBasicBlock, layers, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mbl = MultiBiasedLinear(self.fc.in_features, 180, datasets_len)
        self.fc = nn.Sequential()
        self.lcm = LCM(datasets_len)

    def forward(self, x, d, return_PagIx=False):
        x = super().forward(x)
        x = self.mbl(x, d)
        x = self.lcm(x, d, return_PagIx=return_PagIx)
        return x


def se_resnet18_lcm(datasets_len, **kwargs):
    return SENetLCM(datasets_len, [2, 2, 2, 2], model_name='se_resnet18lcm', **kwargs)


