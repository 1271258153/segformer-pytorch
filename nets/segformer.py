# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class CoordAtt(nn.Module):
    """
    Coordinate Attention (CVPR 2021).
    在通道注意力的基础上引入水平/垂直方向的精确位置信息，
    使网络能够同时关注通道与空间维度的长距离依赖关系。
    """
    def __init__(self, in_channels, reduction=32):
        super(CoordAtt, self).__init__()
        # 自适应池化分别沿 H 和 W 方向聚合信息
        self.x_pool = nn.AdaptiveAvgPool2d((None, 1))   # 沿 W 池化 -> (C, H, 1)
        self.y_pool = nn.AdaptiveAvgPool2d((1, None))   # 沿 H 池化 -> (C, 1, W)

        mid_channels = max(in_channels // reduction, 8)

        # 共享的 1x1 卷积用于降维并融合两方向信息
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels)
        self.act   = nn.SiLU(inplace=True)

        # 分别为 H 方向和 W 方向生成注意力权重
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape

        # 沿 W 方向池化得到 (N, C, H, 1)；沿 H 方向池化得到 (N, C, 1, W) 后转置为 (N, C, W, 1)
        x_h = self.x_pool(x)
        x_w = self.y_pool(x).permute(0, 1, 3, 2)

        # 沿 H 维度拼接两个方向的特征 -> (N, C, H+W, 1)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        # 沿 H 维度拆分回 H 和 W 两个方向
        x_h, x_w = torch.split(y, [h, w], dim=2)
        # 将 W 方向还原为 (N, C, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成两个方向的注意力门控，并利用广播与原特征相乘
        g_h = torch.sigmoid(self.conv_h(x_h))   # (N, C, H, 1)
        g_w = torch.sigmoid(self.conv_w(x_w))   # (N, C, 1, W)

        return identity * g_h * g_w


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        # 在多尺度特征融合后引入 CoordAtt，增强对空间位置与通道信息的联合建模
        self.coord_att   = CoordAtt(embedding_dim, reduction=32)

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # 融合特征经过坐标注意力，强化空间-通道联合建模
        _c = self.coord_att(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

class SegFormer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0', pretrained = False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
