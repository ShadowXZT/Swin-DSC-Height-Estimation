from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy

import numpy as np

N_clusters = [15,  6,  8,  2,  9,  5,  5,  4,  6,  6, 14,  5, 13,  8,  8,  0,  8, 14,
         0,  5,  6, 15,  5,  0, 13, 13,  5, 12,  0,  8,  5,  8,  6,  8,  8, 15,
         7, 12,  6,  2,  5, 11,  6, 14,  8, 15,  0, 14,  8, 13, 13,  9, 14, 15,
        13,  1,  8, 13,  2,  7,  9, 15, 13,  5, 15,  8, 13, 15,  4,  6, 13,  8,
        13, 15, 15, 14,  6,  4,  2,  8, 13,  8, 12, 13,  0, 11, 13,  2,  1, 15,
         6, 14,  5, 14,  5,  6, 15,  5,  1, 15,  1,  4,  8, 15,  5, 10, 13,  6,
         2,  9,  8, 14, 14, 14, 14,  8,  8, 14, 13,  6,  5, 14,  8, 15,  5, 13,
         8,  7, 15,  5,  7,  5, 11,  8, 15,  7, 14, 15,  2, 15,  5,  2, 13,  8,
        14,  6,  5,  5,  8, 14,  5,  8, 13,  5, 13,  5, 13,  8, 13,  2,  0,  5,
         6,  5, 14,  9,  8, 13,  2,  8,  8, 12, 14,  8,  9,  2,  8,  8,  2, 14,
         8,  0,  5, 14,  9, 14,  5,  5,  9,  8, 14, 10,  8,  9, 15,  9,  2, 12,
         8,  2,  9, 12, 13,  5,  5,  2,  5,  5, 12, 13,  4,  2, 13,  6,  1,  5,
        13,  5,  0, 15,  5, 14,  8,  1, 14,  8,  8,  9, 13, 11,  8, 13,  2,  0,
         5,  0,  1,  8,  4, 15,  5, 13,  5, 14,  9,  0, 14,  5,  6,  8, 15,  1,
         8, 14,  0,  5, 13,  4, 14,  7, 13,  8, 15,  2,  1,  2, 10,  0, 13,  1,
         6, 15, 10, 14,  9, 14, 13, 13, 15,  5,  1,  8,  7,  2,  5,  1,  9, 15,
         8, 11,  4,  9,  0,  6,  1,  2,  8,  2,  4,  8,  9,  9,  1, 15,  1,  8,
         8,  8, 13, 13,  1,  0, 13,  2,  9,  6,  8,  8, 14,  5,  8, 13, 15,  2,
        15,  8,  0,  4, 14, 14, 12, 15, 14,  2,  1, 14,  9,  5,  9, 11,  8, 14,
         8,  8,  5,  9,  7, 13,  5,  2, 15, 13,  0,  2, 15,  6, 15,  8,  2,  2,
         2,  8, 15,  8, 14,  4, 14,  9,  8,  8,  2,  7,  5,  9,  1,  3, 15,  8,
         2,  8,  9,  5,  3,  5,  1,  9, 14, 15,  1, 13,  5,  9,  8,  9, 14, 15,
         4, 14, 14, 15,  6, 15, 15, 13,  9, 14,  8, 13,  4,  7, 13,  6,  8,  0,
         6,  8,  6,  0, 13,  5,  2,  9,  2, 14,  9,  6,  8, 14,  9,  7,  8,  7,
        14,  2, 13, 13,  1, 14, 14,  8, 15,  1,  2,  7,  2,  5, 13,  8,  4,  6,
        10,  1, 14,  0,  2,  8,  5, 11, 13,  5,  8,  8,  0,  2,  0,  0,  8,  6,
        13,  8, 14,  5, 11, 13, 14,  6,  6,  6,  8, 13,  7,  6,  2,  3, 15,  1,
         5,  8, 13,  2, 13,  5, 10,  5,  8, 15,  9, 13,  4,  1, 14,  9,  4,  4,
         9,  7,  0,  6, 13, 13,  8, 13]

N_clusters = torch.Tensor(N_clusters).cuda().long()

class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        #self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        #normal_init(self.conv_seg, mean=0, std=0.01)
        pass

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #seg_logits,  = self.forward(inputs)
        losses_r = dict()
        losses = dict()
        mseg_logits = self.forward(inputs)

        if isinstance(mseg_logits, tuple):
            seg_logits, ori_output, distill_output_mu, distill_output_sigma = mseg_logits[0], mseg_logits[1], mseg_logits[2], mseg_logits[3]
            losses = self.losses(seg_logits, gt_semantic_seg)
            #losses_r = self.distillation_loss(distill_output_mu, ori_output, distill_output_sigma)
            losses_r = self.losses_distill(distill_output_mu, ori_output, distill_output_sigma)

        else:
            seg_logits = mseg_logits
            losses = self.losses(seg_logits, gt_semantic_seg)

        losses = {**losses, **losses_r}
        #losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        multi_scale_seg_logits = self.forward(inputs)
        output = multi_scale_seg_logits

        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        if True:
            seg_logit = resize(
                input=seg_logit,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        
        seg_label = seg_label.squeeze(1)

        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            #weight=seg_weight,
            ignore_index=self.ignore_index)
        return loss


    # adapted from https://github.com/clovaai/overhaul-distillation/blob/master/Segmentation/distiller.py
    def distillation_loss(self, source, target, margin=0.001):
        #noise_maps = torch.sample(nsigma)
        #source = source + noise_maps
        loss_distill = dict()
        trans_targets = target.clone()
        for it in range(16):
            trans_target = target[:,N_clusters==it,...].mean(1) # 1,...
            trans_targets[:,it,...] = trans_target
        trans_targets = trans_targets[:,:16,...]
        margin = torch.ones_like(trans_targets).cuda()*margin
        target = torch.max(trans_targets, margin)
        assert target.shape[1] == source.shape[1]
        loss = torch.nn.functional.mse_loss(source, target, reduction="none")
        loss = loss * ((source > target) | (target > 0)).float()
        loss_distill['loss_distill'] = loss.sum() * 0.1
        return loss_distill

    def losses_distill(self, source, target, distill_sigma):
        #KLD loss
        loss_distill = dict()
        trans_targets = target.clone()
        '''
        loss_recons = 0
        N,_,H,W = trans_targets.shape
        sampled_outs = torch.zeros([N,512,H,W]).cuda()
        for i in range(512):
            latent_idx = N_clusters[i]
            sampled_outs[:,i,...] = source[:,latent_idx,...] + distill_sigma[:,latent_idx,...] * torch.randn(distill_sigma[:,latent_idx,...].size()).cuda()
        loss_recons = 0.5 * torch.nn.functional.mse_loss(sampled_outs, target, reduction="mean")
        loss_distill['loss_recons'] = loss_recons
        #'''
        for it in range(16):
            trans_target = target[:,N_clusters==it,...].mean(1) # 1,...
            trans_targets[:,it,...] = trans_target
        trans_targets = trans_targets[:,:16,...]
        margin = torch.ones_like(distill_sigma).cuda()*0.001
        #distill_sigma = torch.exp(0.5 * distill_sigma)
        distill_sigma = torch.max(distill_sigma, margin)
        distill_sigma = torch.clip(distill_sigma, -5, 5)
        p_sigma_pow = distill_sigma * distill_sigma
        q_sigma_pow = 1
        p_mu = source
        q_mu = trans_targets
        kl_loss = torch.mean(-0.5 * (1 + (p_sigma_pow/q_sigma_pow).log() - (p_sigma_pow/q_sigma_pow) - (p_mu-q_mu)*(p_mu-q_mu)/q_sigma_pow).sum(), dim=0)
        loss_distill['loss_distill'] = kl_loss.sum() * 0.0
        return loss_distill
