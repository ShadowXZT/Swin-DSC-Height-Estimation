import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from segmentation_models_pytorch.base.modules import Flatten, Activation
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head_flow import BaseDecodeHead
from .psp_head import PPM

import pdb

class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=1, pad=0):
        super(DUpsampling, self).__init__()
        ## W matrix
        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding = pad,bias=False)
        ## P matrix
        #self.conv_p = nn.Conv2d(num_class * scale * scale, inplanes, kernel_size=1, padding = pad,bias=False)

        self.scale = scale
    
    def forward(self, x):
        x = self.conv_w(x)
        N, C, H, W = x.size()

        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1) 

        # N, W, H*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))

        # N, H*scale, W, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, H*scale, W*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view((N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N, C/(scale**2), H*scale, W*scale
        x = x_permuted.permute(0, 3, 1, 2)
        
        return x


class RegressionHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(RegressionHead, self).__init__()
        self.up_output = DUpsampling(in_channels, 4)
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=kernel_size // 2
        )
    
    def forward(self, inputx):
        x = self.conv2d(inputx)
        x = self.up_output(x)
        return x


class EncoderRegressionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels=2, kernel_size=3):
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=0.5, inplace=True)
        linear = nn.Linear(in_channels, 2, bias=True)
        super().__init__(pool, flatten, dropout, linear)


class ScaleHead(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.flatten = torch.flatten
        self.dot = torch.dot

    def forward(self, mag, height):
        curr_mag = self.flatten(mag, start_dim=1)
        curr_height = self.flatten(height, start_dim=1)
        batch_size = curr_mag.shape[0]
        length = curr_mag.shape[1]
        denom = (
            torch.squeeze(
                torch.bmm(
                    curr_height.view(batch_size, 1, length),
                    curr_height.view(batch_size, length, 1),
                )
            )
            + 0.01
        )
        pinv = curr_height / denom.view(batch_size, 1)
        scale = torch.squeeze(
            torch.bmm(
                pinv.view(batch_size, 1, length), curr_mag.view(batch_size, length, 1)
            )
        )
        return scale

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

        self.scale_out = ScaleHead()
        self.xydir_out = EncoderRegressionHead(self.channels)
        self.agl_out = RegressionHead(self.channels, 1)
        self.mag_out = RegressionHead(self.channels, 1)
        self.fpn_convs = nn.ModuleList()
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

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        fpn_outs = torch.cat(fpn_outs, dim=1)

        output = self.fpn_bottleneck(fpn_outs)

        xydir = self.xydir_out(output)

        agl = self.agl_out(output)

        mag_output = self.mag_out(output)

        #print(agl.shape)
        #print(mag_output.shape)
        
        scale = self.scale_out(mag_output, agl)

        #TODO output the small feature map
        return (mag_output, scale, xydir, agl)
