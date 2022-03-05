"""
Author: Isabella Liu 8/8/21
Feature: A simple, shallow discriminator network
"""

import math
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, inplane, outplane):
        super(Discriminator, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv3d(inplane, 32, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            #nn.LeakyReLU(0.2, True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            #nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            #nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
            #nn.LeakyReLU(0.2, True)
            #nn.LeakyReLU(0.2, inplace=True)
        )
        self.net2 = nn.Sequential(
            nn.Conv3d(128, 1, kernel_size=3, stride=1, padding=1, dilation=1),
            #nn.Linear(128, outplane),
            nn.Sigmoid()
            #nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        """
        :param input: [bs, 1, patch_H, patch_W] TODO optional patch size, here patch size is hard coded as 32
        :return: [bs, 1] float between 0 and 1
        """
        bs = input.shape[0]
        output = self.net1(input)
        #print('before ', output.shape)
        #output = torch.permute(output, (0,2,3,4,1))
        #print('before ', output.shape)
        output = self.net2(output)
        #output = self.net3(output).view(bs, 1)
        return output


class SimpleD64(nn.Module):
    def __init__(self):
        super(SimpleD64, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=4, padding=1, dilation=1),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=4, padding=1, dilation=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        :param input: [bs, 1, patch_H, patch_W] TODO optional patch size, here patch size is hard coded as 64
        :return: [bs, 1] float between 0 and 1
        """
        bs = input.shape[0]
        output = self.net1(input)
        output = self.net2(output)
        output = self.net3(output).view(bs, 1)
        return output


if __name__ == '__main__':
    discriminator32 = SimpleD32().cuda()
    discriminator64 = SimpleD64().cuda()
    input = torch.rand(2, 1, 64, 64).cuda()
    output = discriminator64(input)
    print(output)