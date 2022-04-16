"""
Author: Isabella Liu 4/28/21
Feature: Hourglass and PSMNet (stacked hourglass) module
"""

import math

from .psmnet_submodule_ray import *
from torch.autograd import Variable


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

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 ):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.ReLU(inplace=True)
        # self.bn = nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(x))

class CostRegNet(nn.Module):
    def __init__(self, in_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            nn.BatchNorm3d(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            nn.BatchNorm3d(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            nn.BatchNorm3d(8))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        # x = self.conv12(x)
        return x


class PSMNet(nn.Module):
    def __init__(self, maxdisp=192, transform=False):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.transform = transform
        self.feature_extraction = FeatureExtraction(transform)

        self.down = 2
        self.cost_reg = CostRegNet(64)


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
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False)
        )
        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False)
        )
        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False)
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

        #print(cost.shape)
        """
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        #print(out1.shape, cost0.shape)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2
        """
        cost = F.interpolate(cost, (self.maxdisp//4, 4 * H, 4 * W), mode='trilinear', align_corners=False)
        
        cost = self.cost_reg(cost)

        if self.training:
            return cost
        else:
            return cost

class Renderer(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=1, input_ch_feat=8, skips=[4], n_rays=1024):
        """
        """
        super(Renderer, self).__init__()
        self.D = D
        self.W = W
        self.n_rays = n_rays
        self.input_ch = input_ch
        #self.input_ch_views = input_ch_views
        self.skips = skips
        #self.use_viewdirs = use_viewdirs
        self.in_ch_pts, self.in_ch_feat = input_ch, input_ch_feat

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_ch_pts, W, bias=True)] + [nn.Linear(W, W, bias=True) if i not in self.skips else nn.Linear(W + self.in_ch_pts, W) for i in range(D-1)])
        self.pts_bias = nn.Linear(input_ch_feat, W)
        #self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])


        #self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)

        #self.output_linear = nn.Linear(W, output_ch)

        self.pts_linears.apply(weights_init)
        #self.views_linears.apply(weights_init)
        #self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)

        self.ray_reg = RayRegression()

        #self.rgb_linear.apply(weights_init)

    def forward(self, x, disp_candidate,B,D):

        dim = x.shape[-1]
        #in_ch_feat = dim-self.in_ch_pts
        input_pts, input_feats = torch.split(x, [self.in_ch_pts, self.in_ch_feat], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = torch.relu(self.alpha_linear(h))
        output = alpha.view(B,D,args.n_rays,1)
        pred_disp = rayreg_model(output, disp_candidate)
        return pred_disp

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


if __name__ == '__main__':
    # Test PSMNet
    model = PSMNet(maxdisp=192).cuda()
    model.eval()
    img_L = torch.rand(1, 3, 256, 512).cuda()
    img_R = torch.rand(1, 3, 256, 512).cuda()
    pred = model(img_L, img_R)
    print(f'pred shape {pred.shape}')   # pred shape torch.Size([1, 1, 256, 512])