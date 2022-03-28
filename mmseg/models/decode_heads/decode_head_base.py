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

        batch_size = 800
        seg_label = torch.ones([1,512,512])

        self.x_A = torch.randint(0, seg_label.shape[1], (batch_size,))
        self.y_A = torch.randint(0, seg_label.shape[2], (batch_size,))

        self.x_B = torch.randint(0, seg_label.shape[1], (batch_size,))
        self.y_B = torch.randint(0, seg_label.shape[2], (batch_size,))

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
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
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

    '''def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output'''

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
        
        #np.save('/home/xshadow/Swin-Transformer-Semantic-Segmentation/height_out/depth_out_{}.npy'.format(str(1)),
        #         seg_logit[0].data.cpu().numpy()) 
        seg_label = seg_label.squeeze(1)
        #loss['midas_loss_seg'] = 0.5 * self.gradient_loss(seg_logit, seg_label, 4)
        #loss['relative_loss_seg'] = self.losses_relative_constraint(seg_logit, seg_label, self.x_A, self.y_A, self.x_B, self.y_B)

        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            #weight=seg_weight,
            ignore_index=self.ignore_index)
        return loss


    def losses_relative_constraint(self, seg_logit, seg_label, x_A, y_A, x_B, y_B):
        """Compute segmentation loss."""
        batch_size = 800
        n_total = seg_logit.shape[2] * seg_logit.shape[3]

        N, C, H, W = seg_logit.shape
        seg_logit = seg_logit.squeeze(1)
        total_loss = 0

        for ind in range(N):
            #x_A = torch.randint(0, seg_label.shape[1], (batch_size,))
            x_A = ((x_A/512.)*seg_label.shape[1]).long().cuda()
            #y_A = torch.randint(0, seg_label.shape[2], (batch_size,))
            y_A = ((y_A/512.)*seg_label.shape[2]).long().cuda()
            #x_B = torch.randint(0, seg_label.shape[1], (batch_size,))
            #y_B = torch.randint(0, seg_label.shape[2], (batch_size,))
            x_B = ((x_B/512.)*seg_label.shape[1]).long().cuda()
            y_B = ((y_B/512.)*seg_label.shape[2]).long().cuda()
            z_A = seg_logit[ind][x_A,y_A]
            z_B = seg_logit[ind][x_B,y_B]
            g_A = seg_label[ind][x_A,y_A]
            g_B = seg_label[ind][x_B,y_B]

            ground_truth = torch.zeros_like(g_A).cuda()
            ground_truth[g_A - g_B >= 0.1] = 1
            ground_truth[g_A - g_B <= -0.1] = -1
            mask = torch.abs(ground_truth)
            relative = -ground_truth*(z_A-z_B)
            relative = torch.clip(relative, -50, 50)
            lossr = mask * torch.log(1 + torch.exp(relative)) \
                                               + (1 - mask) * (z_A - z_B) * (z_A - z_B)

            loss = torch.sum(lossr) / batch_size
            total_loss += loss

        return total_loss


    def gradient_loss(self, prediction, target, scales=1):

        total = 0
        prediction = prediction.squeeze()

        for scale in range(scales):
            step = pow(2,scale)
            pred = prediction[:, ::step, ::step]
            tget = target[:, ::step, ::step]
            n_pixels = tget.shape[1] * tget.shape[2]
            diff = pred - tget
            grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
            grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
            image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
            total += torch.mean(image_loss) / n_pixels

        return total


    def mssi_loss(self, preds, depth):
        n_pixels = depth.shape[1] * depth.shape[2]
        d = torch.squeeze(preds) - depth
        grad_loss_term = self.gradient_loss(preds, depth, 1)
        term_1 = torch.pow(d.view(-1, n_pixels),2).mean(dim=1).sum() #pixel wise mean, then batch sum
        term_2 = (torch.pow(d.view(-1, n_pixels).sum(dim=1),2)/(2*(n_pixels**2))).sum()
        bloss = term_1 - 0.5 * term_2 + 0.1 * grad_loss_term
        return torch.mean(bloss)

    def midas_loss(self, preds, depth):
        n_pixels = depth.shape[1] * depth.shape[2]
        d = preds - depth
        grad_loss_term = self.gradient_loss(d, n_pixels, 4)
        mse_loss = torch.mean(d * d)

        return mse_loss + grad_loss_term
