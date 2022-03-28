import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from modules import PSDeformConv, _PSDeformConv, PSDeformConvPack

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM

import numpy as np

import pdb

global NUM

NUM = 0

class RegressionHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(RegressionHead, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, 1, kernel_size
        )

    def forward(self, inputx):
        x = resize(inputx,size=(inputx.shape[2]*4, inputx.shape[3]*4), mode='bilinear', align_corners=False)
        x = self.conv2d(x)
        return x


@HEADS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.max_linear_out = nn.Linear(self.channels, 1)
        self.ps_conv = PSDeformConvPack(self.channels, self.channels, kernel_size=3, stride=1, padding=1)
        self.norm_depth_out = RegressionHead(self.channels, 1)
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])
        low_resolution_depth = fpn_outs[-1]

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        
        output = self.fpn_bottleneck(fpn_outs)
        asc, output = self.ps_conv(output)
        global NUM
        #asc = asc.cpu().numpy().astype(np.float16)
        #np.save('./ahn_ps_maxnorm_ascs/out_'+str(NUM)+'.npy', asc)
        NUM += 1

        #pool_output = torch.mean(output, (2,3))
        pool_output = F.max_pool2d(output, kernel_size=output.size()[2:])
        
        pool_output = pool_output.squeeze()

        routput = self.norm_depth_out(output)

        max_output = self.max_linear_out(pool_output)

        output = routput * max_output.view(routput.shape[0],1, 1,1)

        return (output, routput, max_output)
