import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import *
from torchvision import models


class Model(nn.Module):
    def __init__(self, n_channels=2560, bilinear=False):
        super(Model, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        # self.batchNorm = batchNorm
        self.conv1 = (DoubleConv(n_channels, n_channels)) #2560 -> 2560
        self.conv2 = (DoubleConv(n_channels, n_channels*2)) #2560 ->2560*2
        factor = 2 if bilinear else 1
        self.conv3 = (DoubleConv(n_channels*2, n_channels*2 // factor)) #2560*2 ->2560*2
        self.up1 = (Up(n_channels*2 // factor, n_channels // factor, bilinear))
        self.conv4 = (DoubleConv(n_channels // factor, n_channels // factor))
        self.up2 = (Up(n_channels // factor, n_channels // (2 * factor), bilinear))
        self.conv5 = (DoubleConv(n_channels // (2 * factor), n_channels // (2 * factor)))
        self.up3 = (UpConcat(n_channels // (2 * factor), n_channels // (4 * factor)))
        self.conv6 = (DoubleConv(n_channels // (4 * factor), 256))
        self.conv7 = UpSample(256, 256)
        self.conv8 = UpSample(256, 256)
        self.conv9 = UpSample(256, 256)
        self.conv10 = (DoubleConv(256, 128, 256))
        self.conv11 = (DoubleConv(128, 64, 128))
        self.conv12 = (DoubleConv(64, 64))
        self.outc = (OutConv(64, 2))

    def forward(self, x):
        ## x format: [0, 1, 2, 3]

        x1 = self.conv1(x[0])
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.up1(x3, x[1]) #
        x5 = self.conv4(x4)
        x6 = self.up2(x5, x[2])
        x7 = self.conv5(x6)
        x8 = self.up3(x7, x[3])
        x9 = self.conv6(x8)
        x10 = self.conv7(x9)
        y = self.conv8(x10)
        y = self.conv9(y)
        y = self.conv10(y)
        y = self.conv11(y)
        y = self.conv12(y)
        y = self.outc(y)

        return y
# model = Model()
# # print(model)

# a = torch.randn((1, 2560, 12, 39))
# b = torch.randn((1, 2560, 23, 78))
# c = torch.randn((1, 1280, 46, 155))
# d = torch.randn((1, 640, 46, 155))
# x = [a, b, c, d]
# print(model(x).shape)

# # 1, 2560, 12, 39  1, 2560, 23, 78  1, 1280, 46, 155  1, 640, 46, 155


# # a = torch.randn((1, 1280, 30, 30))
# # b = torch.randn((1, 1280, 60, 60))
# # c = torch.randn((1, 640, 120, 120))
# # d = torch.randn((1, 320, 120, 120))