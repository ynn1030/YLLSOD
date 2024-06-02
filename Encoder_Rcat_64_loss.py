import torch
import torch.nn as nn
import pvt_v2
import torchvision.models as models
from torch.nn import functional as F
from layers import *
import numbers
from NDM_model import Conv2dBlock
import numpy as np
import cv2
import os

class Transformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.encoder = getattr(pvt_v2, backbone)()
        if pretrained:
            checkpoint = torch.load('./pvt_v2_b3.pth', map_location='cpu')
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            state_dict = self.encoder.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            self.encoder.load_state_dict(checkpoint_model, strict=False)

def Encoder():
    model = Transformer('pvt_v2_b3', pretrained=True)
    return model

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_plane, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane * 2, out_plane, kernel_size=1, stride=1, relu=False)

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

##########################################################################
## Layer Norm

def to_3d(x):
    # return rearrange(x, 'b c h w -> b (h w) c')
    b, c, h, w = x.shape
    return x.view(b, c, -1).transpose(1, 2).contiguous()


def to_4d(x, h, w):
    # return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
    b, t, c = x.shape
    return x.transpose(1, 2).contiguous().view(b, c, h, w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))  # (b,c,h,w)->(b,3*c,h,w)->(b,3*c,h,w)
        q, k, v = qkv.chunk(3, dim=1)  # (b,3*c,h,w)->(b,c,h,w),(b,c,h,w),(b,c,h,w)

        q = q.view(q.shape[0], self.num_heads, q.shape[1] // self.num_heads, q.shape[2] * q.shape[3])
        k = k.view(k.shape[0], self.num_heads, k.shape[1] // self.num_heads, k.shape[2] * k.shape[3])
        v = v.view(v.shape[0], self.num_heads, v.shape[1] // self.num_heads, v.shape[2] * v.shape[3])

        q = torch.nn.functional.normalize(q, dim=-1)  # 对最后一维进行normalize
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = out.view(out.shape[0], out.shape[1] * out.shape[2], h, w)
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

#########################################################################

#########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   ###多头自注意力 MDTA
        x = x + self.ffn(self.norm2(x))    ###门控前馈网络 GDFN

        return x
#########################################################################

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4, x8):
        x = torch.cat([x1, x2, x4, x8], dim=1)
        return self.conv(x)

class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        base_channel = 32
        model = Encoder()
        self.rgb_net = model.encoder

        ###MIMOUNet
        self.SCM1 = SCM(64)
        self.SCM2 = SCM(128)
        self.SCM3 = SCM(320)
        self.SCM4 = SCM(512)

        self.FAM1 = FAM(64)
        self.FAM2 = FAM(128)
        self.FAM3 = FAM(320)
        self.FAM4 = FAM(512)

        self.feat_extract = nn.ModuleList([
            BasicConv(64, 64, kernel_size=3, relu=True, stride=1),
            BasicConv(64, 128, kernel_size=3, relu=True, stride=2),
            BasicConv(128, 320, kernel_size=3, relu=True, stride=2),
            BasicConv(320, 512, kernel_size=3, relu=True, stride=2),

            BasicConv(512, 320, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(320, 128, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(128, 64, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(64, 1, kernel_size=1, relu=False, stride=1),

        ])
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=64, num_heads=1,
                             ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias') for i in range(1)])
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=128, num_heads=2,
                             ffn_expansion_factor=2.66,bias=False,
                             LayerNorm_type='WithBias') for i in range(1)])
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=320, num_heads=2,
                             ffn_expansion_factor=2.66,bias=False,
                             LayerNorm_type='WithBias') for i in range(1)])
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=512, num_heads=4,
                             ffn_expansion_factor=2.66,bias=False,
                             LayerNorm_type='WithBias') for i in range(1)])

        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=512, num_heads=4,
                             ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias') for i in range(1)])
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=320, num_heads=2,
                             ffn_expansion_factor=2.66,bias=False,
                             LayerNorm_type='WithBias') for i in range(1)])
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=128, num_heads=2,
                             ffn_expansion_factor=2.66,bias=False,
                             LayerNorm_type='WithBias') for i in range(1)])
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=64, num_heads=1,
                             ffn_expansion_factor=2.66,bias=False,
                             LayerNorm_type='WithBias') for i in range(1)])
        self.AFFs = nn.ModuleList([
            AFF(1024, 64),
            AFF(1024, 128),
            AFF(1024, 320),
            AFF(1024, 512),
        ])
        self.Convs = nn.ModuleList([
            BasicConv(640, 320, kernel_size=1, relu=True, stride=1),
            BasicConv(256, 128, kernel_size=1, relu=True, stride=1),
            BasicConv(128, 64, kernel_size=1, relu=True, stride=1),
        ])
        self.output = nn.Conv2d(64, 1, 1, 1, 0)


        self.conv0 = Conv2dBlock(4, 32, 3, 1, 1, norm='none', activation='relu', pad_type='reflect')
        self.conv1 = Conv2dBlock(4, 64, 9, 1, 4, norm='none', activation='none', pad_type='reflect')
        self.conv2 = Conv2dBlock(64, 64, 3, 1, 1, norm='none', activation='relu', pad_type='reflect')
        self.conv3 = Conv2dBlock(64, 128, 3, 2, 1, norm='none', activation='relu', pad_type='reflect')
        self.conv4 = Conv2dBlock(128, 128, 3, 1, 1, norm='none', activation='relu', pad_type='reflect')
        self.conv5 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.activation = nn.ReLU(inplace=True)
        self.conv6 = Conv2dBlock(128, 64, 3, 1, 1, norm='none', activation='relu', pad_type='reflect')
        self.conv7 = Conv2dBlock(96, 64, 3, 1, 1, norm='none', activation='none', pad_type='reflect')
        self.conv8 = Conv2dBlock(64, 4, 3, 1, 1, norm='none', activation='none', pad_type='reflect')

        self.conv_input = Conv2dBlock(6, 3, 3, 1, 1, norm='none', activation='relu', pad_type='reflect')


    def forward(self, imgs):


        input_max = torch.max(imgs, dim=1, keepdim=True)[0]
        image = torch.cat((input_max, imgs), dim=1)

        x0 = self.conv0(image)
        # print('x0:', x0.shape)
        x1 = self.conv1(image)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = self.activation(x5)
        cat5 = torch.cat((x5, x2), dim=1)
        x6 = self.conv6(cat5)
        cat6 = torch.cat((x6, x0), dim=1)
        x7 = self.conv7(cat6)
        x8 = self.conv8(x7)

        R = torch.sigmoid(x8[:, 0:3, :, :])
        L = torch.sigmoid(x8[:, 3:4, :, :])


        input_cat = self.conv_input(torch.cat((imgs, R), dim=1))

        img1, img2, img3, img4 = self.rgb_net(input_cat)
        img1_ = self.SCM1(img1)
        img2_ = self.SCM2(img2)
        img3_ = self.SCM3(img3)
        img4_ = self.SCM4(img4)

        l1_1 = self.feat_extract[0](img1_)
        l1_2 = self.FAM1(l1_1, img1_)
        tfen1 = self.encoder_level1(l1_2)

        l2_1 = self.feat_extract[1](tfen1)
        l2_2 = self.FAM2(l2_1, img2_)
        tfen2 = self.encoder_level2(l2_2)

        l3_1 = self.feat_extract[2](tfen2)
        l3_2 = self.FAM3(l3_1, img3_)
        tfen3 = self.encoder_level3(l3_2)

        l4_1 = self.feat_extract[3](tfen3)
        l4_2 = self.FAM4(l4_1, img4_)
        tfen4 = self.encoder_level4(l4_2)

        tfen1_2down = F.interpolate(tfen1, scale_factor=0.5)
        tfen1_4down = F.interpolate(tfen1_2down, scale_factor=0.5)
        tfen1_8down = F.interpolate(tfen1_4down, scale_factor=0.5)

        tfen2_2up = F.interpolate(tfen2, scale_factor=2)
        tfen2_2down = F.interpolate(tfen2, scale_factor=0.5)
        tfen2_4down = F.interpolate(tfen2_2down, scale_factor=0.5)

        tfen3_2up = F.interpolate(tfen3, scale_factor=2)
        tfen3_4up = F.interpolate(tfen3_2up, scale_factor=2)
        tfen3_2down = F.interpolate(tfen3, scale_factor=0.5)

        tfen4_2up = F.interpolate(tfen4, scale_factor=2)
        tfen4_4up = F.interpolate(tfen4_2up, scale_factor=2)
        tfen4_8up = F.interpolate(tfen4_4up, scale_factor=2)

        cat96 = self.AFFs[0](tfen1, tfen2_2up, tfen3_4up, tfen4_8up)
        cat48 = self.AFFs[1](tfen1_2down, tfen2, tfen3_2up, tfen4_4up)
        cat24 = self.AFFs[2](tfen1_4down, tfen2_2down, tfen3, tfen4_2up)
        cat12 = self.AFFs[3](tfen1_8down, tfen2_4down, tfen3_2down, tfen4)

        tfde4 = self.decoder_level4(cat12)
        l_4_1 = self.feat_extract[4](tfde4)

        l_3_cat = torch.cat([l_4_1, cat24], dim=1)
        l_3_cat = self.Convs[0](l_3_cat)
        tfde3 = self.decoder_level3(l_3_cat)
        l_3_1 = self.feat_extract[5](tfde3)

        l_2_cat = torch.cat([l_3_1, cat48], dim=1)
        l_2_cat = self.Convs[1](l_2_cat)
        tfde2 = self.decoder_level2(l_2_cat)
        l_2_1 = self.feat_extract[6](tfde2)

        l_1_cat = torch.cat([l_2_1, cat96], dim=1)
        l_1_cat = self.Convs[2](l_1_cat)
        tfde1 = self.decoder_level1(l_1_cat)

        l_1_1 = self.feat_extract[7](tfde1)
        output1 = F.interpolate(l_1_1, (384, 384), mode='bilinear', align_corners=True)
        output = F.sigmoid(output1)

        return output1, output, R, L



