"""
Author: Isabella Liu 4/28/21
Feature: Hourglass and PSMNet (stacked hourglass) module
"""

import math

from .psmnet_submodule import *


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(
            convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes)
        )

    def forward(self, x, presqu, postqu):
        out = self.conv1(x)
        pre = self.conv2(out)
        if postqu is not None:
            pre = F.relu(pre + postqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)
        out = self.conv4(out)

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)
        return out, pre, post


class PSMNet(nn.Module):
    def __init__(self, maxdisp=192, loss='BCE', transform=False):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.loss = loss
        self.transform = transform
        self.feature_extraction = FeatureExtraction(transform)

        self.dres0 = nn.Sequential(
            convbn_3d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.dres1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1)
        )

        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )
        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )
        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        # for m in self.modules():
        #     if type(m) == (nn.Conv2d or nn.Conv3d or nn.Linear) :
        #         torch.nn.init.xavier_uniform_(m.weight)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, img_L, img_R, img_L_transformed=None, img_R_transformed=None):

        if self.transform:
            refimg_feature = self.feature_extraction(img_L, img_L_transformed)  # [bs, 32, H/4, W/4]
            targetimg_feature = self.feature_extraction(img_R, img_R_transformed)
        else:
            refimg_feature = self.feature_extraction(img_L)  # [bs, 32, H/4, W/4]
            targetimg_feature = self.feature_extraction(img_R)

        # Cost Volume
        [bs, feature_size, H, W] = refimg_feature.size()
        cost = torch.FloatTensor(bs, feature_size * 2, self.maxdisp // 4, H, W).zero_().cuda()

        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :feature_size, i, :, i:] = refimg_feature[:, :, :, i:]
                cost[:, feature_size:, i, :, i:] = targetimg_feature[:, :, :, :-i]
            else:
                cost[:, :feature_size, i, :, :] = refimg_feature
                cost[:, feature_size:, i, :, :] = targetimg_feature
        cost = cost.contiguous()  # [bs, fs*2, max_disp/4, H/4, W/4]

        #print(cost.shape)
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            # cost1 = F.upsample(cost1, [self.maxdisp,img_L.size()[2],img_L.size()[3]], mode='trilinear')
            # cost2 = F.upsample(cost2, [self.maxdisp,img_L.size()[2],img_L.size()[3]], mode='trilinear')
            cost1 = F.interpolate(cost1, (self.maxdisp, 4 * H, 4 * W), mode='trilinear', align_corners=False)
            cost2 = F.interpolate(cost2, (self.maxdisp, 4 * H, 4 * W), mode='trilinear', align_corners=False)

            cost1 = torch.squeeze(cost1, 1)
            cost1 = F.softmax(cost1, dim=1)
            pred1 = DisparityRegression(self.maxdisp)(cost1)

            cost2 = torch.squeeze(cost2, 1)
            cost2 = F.softmax(cost2, dim=1)
            pred2 = DisparityRegression(self.maxdisp)(cost2)

        # cost3 = F.upsample(cost3, [self.maxdisp,img_L.size()[2],img_L.size()[3]], mode='trilinear')
        cost3 = F.interpolate(cost3, (self.maxdisp, 4 * H, 4 * W), mode='trilinear', align_corners=False)
        cost3 = torch.squeeze(cost3, 1)
        cost3 = F.softmax(cost3, dim=1)

        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        #print(cost3.shape)
        pred3 = DisparityRegression(self.maxdisp)(cost3)

        if self.training and self.loss == 'BCE':
            return pred1, pred2, pred3, cost1, cost2, cost3
        elif self.training:
            return pred1, pred2, pred3
        else:
            return pred3, cost3


if __name__ == '__main__':
    # Test PSMNet
    model = PSMNet(maxdisp=192).cuda()
    model.eval()
    img_L = torch.rand(1, 3, 256, 512).cuda()
    img_R = torch.rand(1, 3, 256, 512).cuda()
    pred = model(img_L, img_R)
    print(f'pred shape {pred.shape}')   # pred shape torch.Size([1, 1, 256, 512])