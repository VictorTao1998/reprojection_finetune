"""
Author: Isabella Liu 8/8/21
Feature: A simple, shallow discriminator network
"""

import math
import torch
import torch.nn as nn

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, inplane, outplane):
        super(Discriminator, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv3d(inplane, 32, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            #nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1)
            nn.LeakyReLU(0.2, True)
            #nn.LeakyReLU(0.2, inplace=True)
        )
        self.net2 = nn.Sequential(
            #nn.Conv3d(128, 1, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Linear(128, outplane),
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
        output = torch.permute(output, (0,2,3,4,1))
        #print('before ', output.shape)
        output = self.net2(output)
        #output = self.net3(output).view(bs, 1)
        return output

class Discriminator2(nn.Module):
    def __init__(self, inplane, outplane):
        super(Discriminator2, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv3d(inplane, 32, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            #nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1)
            nn.LeakyReLU(0.2, True)
            #nn.LeakyReLU(0.2, inplace=True)
        )
        self.net2 = nn.Sequential(
            #nn.Conv3d(128, 1, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Linear(128, outplane),
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
        output = torch.permute(output, (0,2,3,4,1))
        #print('before ', output.shape)
        output = self.net2(output)
        #output = self.net3(output).view(bs, 1)
        return output

class Critic2(nn.Module):
    def __init__(self, inplane, outplane):
        super(Critic2, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv3d(inplane, 32, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            #nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1)
            nn.LeakyReLU(0.2, True)
            #nn.LeakyReLU(0.2, inplace=True)
        )
        self.net2 = nn.Sequential(
            #nn.Conv3d(128, 1, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Linear(128, outplane)
            #nn.Sigmoid()
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
        output = torch.permute(output, (0,2,3,4,1))
        #print('before ', output.shape)
        output = self.net2(output)
        #output = self.net3(output).view(bs, 1)
        return output

class Critic(nn.Module):
    def __init__(self, inplane, outplane):
        super(Critic, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv3d(inplane, 32, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            #nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1)
            nn.LeakyReLU(0.2, True)
            #nn.LeakyReLU(0.2, inplace=True)
        )
        self.net2 = nn.Sequential(
            #nn.Conv3d(128, 1, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Linear(128, outplane)
            #nn.Sigmoid()
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
        output = torch.permute(output, (0,2,3,4,1))
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