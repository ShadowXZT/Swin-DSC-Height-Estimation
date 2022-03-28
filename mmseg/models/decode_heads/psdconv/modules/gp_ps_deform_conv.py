#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair

import gpytorch


from functions.ps_deform_conv_func import PSDeformConvFunction


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)





class PSDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(PSDeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels//groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset, scale_c):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            scale_c.shape[1]
        return PSDeformConvFunction.apply(input, offset, scale_c,
                                                   self.weight,
                                                   self.bias,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   self.groups,
                                                   self.deformable_groups,
                                                   self.im2col_step)

_PSDeformConv = PSDeformConvFunction.apply

class PSDeformConvPack(PSDeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(PSDeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)

        out_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
                                          
        self.conv_scale = nn.Conv2d(self.in_channels,
                                          2,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
                                          
        self.init_offset_scale()

        grid_bounds=(-10., 10.) 

        self.gp_layer = GaussianProcessLayer(num_dim=in_channles, grid_bounds=grid_bounds)


    def init_offset_scale(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.conv_scale.weight.data.zero_()
        self.conv_scale.bias.data.zero_() - 10.

    def forward(self, input):
        offset = self.conv_offset(input)
        scale_c = self.conv_scale(input)
        scale_c = scale_c.repeat(1, self.kernel_size[0] * self.kernel_size[1], 1, 1)
        scale_c = torch.sigmoid(scale_c) + 1
        return scale_c[:,0,:,:], PSDeformConvFunction.apply(input, offset, scale_c, 
                                                self.weight, 
                                                self.bias, 
                                                self.stride, 
                                                self.padding, 
                                                self.dilation, 
                                                self.groups,
                                                self.deformable_groups,
                                                self.im2col_step)

