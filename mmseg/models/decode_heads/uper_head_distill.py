import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head_distill import BaseDecodeHead
from .psp_head import PPM
import numpy as np
import cv2
from .unit_cluster import neuron_clusters

import pdb

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
        self.unit_clusters = 16
        #self.nsigma = nn.Parameter(torch.zeros(self.unit_clusters, )) #16, 
        self.height_out = RegressionHead(self.unit_clusters, 1)
        
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

        self.distill_mu_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.unit_clusters,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.distill_sigma_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.unit_clusters,
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

        ori_output = self.fpn_bottleneck(fpn_outs) #N, 512, h, w
        distill_output_mu = self.distill_mu_bottleneck(fpn_outs) #N, 16, h, w
        distill_output_sigma = self.distill_sigma_bottleneck(fpn_outs) #N, 16, h, w
        distill_output = distill_output_mu
        #distill_output = distill_output_mu + distill_output_sigma * torch.randn(distill_output_sigma.size()).cuda() * 0.001
        #'''
        for i in range(16):
            simage = distill_output[0,i,...]
            simage = simage.cpu().numpy()
            simage = (simage - simage.min()) / (simage.max() - simage.min() + 1e-6) * 255
            simage = simage.astype(np.uint8)
            #ret, bw_img = cv2.threshold(simage, 100, 255, cv2.THRESH_BINARY)
            ret, bw_img = cv2.threshold(simage, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imwrite('/home/xshadow/Swin-Transformer-Semantic-Segmentation/response_new/bwimg_' + str(i) + '.png', bw_img)
            simage = cv2.applyColorMap(simage, cv2.COLORMAP_JET)
            cv2.imwrite('/home/xshadow/Swin-Transformer-Semantic-Segmentation/response_new/response_' + str(i) + '.png', simage)
            '''if i==7:
                output_cls = torch.from_numpy(bw_img).cuda().float()
                output_cls = output_cls.view([1,1,128,128])
                output_cls = resize(output_cls, size=(512,512), mode='nearest')
                break'''
        pdb.set_trace()
        #'''

        output = self.height_out(distill_output)  #N, 1, h, w

        return output
        #return output, ori_output, distill_output_mu, distill_output_sigma
