import torch
import torch.nn as nn
import torch.nn.functional as F
from dcn_v2 import PCDAlignv2


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + 1,
                              bias=True, dilation=2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class DRDB(nn.Module):
    def __init__(self, nChannels, denseLayer, growthRate):
        super(DRDB, self).__init__()
        num_channels = nChannels
        modules = []
        for i in range(denseLayer):
            modules.append(make_dilation_dense(num_channels, growthRate))
            num_channels += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(num_channels, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class Pyramid(nn.Module):
    def __init__(self, in_channels=6, n_feats=64):
        super(Pyramid, self).__init__()
        self.in_channels = in_channels
        self.n_feats = n_feats
        num_feat_extra = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.n_feats, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        layers = []
        for _ in range(num_feat_extra):
            layers.append(ResidualBlockNoBN())
        self.feature_extraction = nn.Sequential(*layers)

        self.downsample1 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        x_in = self.conv1(x)
        x1 = self.feature_extraction(x_in)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        return [x1, x2, x3]


class SpatialAttentionModule(nn.Module):
    def __init__(self, n_feats):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map


class ADNetv2(nn.Module):
    def __init__(self, nChannel, nDenselayer, nFeat, growthRate, align_version='v0'):
        super(ADNetv2, self).__init__()
        self.n_channel = nChannel
        self.n_denselayer = nDenselayer
        self.n_feats = nFeat
        self.growth_rate = growthRate
        self.align_version = align_version

        # PCD align module
        self.pyramid_feats = Pyramid(3)
        self.align_module = PCDAlignv2()

        # Spatial attention module
        self.att_module_l = SpatialAttentionModule(self.n_feats)
        self.att_module_h = SpatialAttentionModule(self.n_feats)

        # feature extraction
        self.feat_exract = nn.Sequential(
            nn.Conv2d(3, nFeat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True))

        self.fusion = nn.Conv2d(self.n_feats * 6, self.n_feats // 2, kernel_size=3, padding=1, bias=True)
        self.mimo_extractor = MIMOUExtractor()
        self.reconstruction = nn.Sequential(
            nn.Conv2d(self.n_feats // 2, self.n_feats, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_feats, 3, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.LeakyReLU(inplace=True)
    
    def check_image_size(self, x, pad=4):
        _, _, h, w = x.size()
        mod_pad_h = (pad - h % pad) % pad
        mod_pad_w = (pad - w % pad) % pad
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x1, x2=None, x3=None):
        x1_t, x1_l = x1[:,0:3,...], x1[:,3:,...]
        _, _, h_input, w_input = x1_t.size()

        if x2 == None or x3 == None:
            x2_t, x2_l = x1_t.clone(), x1_l.clone()
            x3_t, x3_l = x1_t.clone(), x1_l.clone()
        else:
            x2_t, x2_l = x2[:,0:3,...], x2[:,3:,...]
            x3_t, x3_l = x3[:,0:3,...], x3[:,3:,...]

        # check input size
        x1_t, x1_l = self.check_image_size(x1_t), self.check_image_size(x1_l)
        x2_t, x2_l = self.check_image_size(x2_t), self.check_image_size(x2_l)
        x3_t, x3_l = self.check_image_size(x3_t), self.check_image_size(x3_l)

        
        # pyramid features of linear domain
        f1_l = self.pyramid_feats(x1_l)
        f2_l = self.pyramid_feats(x2_l)
        f3_l = self.pyramid_feats(x3_l)
        f2_ = f2_l[0]
        
        f1_aligned_l = self.align_module(f1_l, f2_l)
        f3_aligned_l = self.align_module(f3_l, f2_l)
        
        # Spatial attention module
        f1_t = self.feat_exract(x1_t)
        f2_t = self.feat_exract(x2_t)
        f3_t = self.feat_exract(x3_t)
        f1_t_A = self.att_module_l(f1_t, f2_t)
        f1_t_ = f1_t * f1_t_A
        f3_t_A = self.att_module_h(f3_t, f2_t)
        f3_t_ = f3_t * f3_t_A

        # fusion subnet
        feat = torch.cat((f1_aligned_l, f1_t_,  f2_, f2_t, f3_aligned_l, f3_t_), 1) 
        feat = self.fusion(feat)
        feat = self.mimo_extractor(feat)
        out = self.reconstruction(feat)
        return out[:, :, :h_input, :w_input]


class MIMOUExtractor(nn.Module):
    def __init__(self, num_res=8):
        super(MIMOUExtractor, self).__init__()
        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(base_channel, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, base_channel, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

    def forward(self, x):
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        z = self.feat_extract[3](z)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z = self.feat_extract[4](z)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)

        return z


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out