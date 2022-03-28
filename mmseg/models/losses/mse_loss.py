import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from ..builder import LOSSES


class AverageMeter(object):
    """Computes and stores the average and current valvalue"""
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size

@LOSSES.register_module()
class MSELoss(nn.Module):
    def __init__(self, use_sigmoid=False, loss_weight=None):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, out, target, ignore_index=-100):
        target = target.float()
        #h, w = target.size(1), target.size(2)
        #out = F.interpolate(out, size=[h, w], mode='bilinear')

        loss = self.criterion(out.squeeze(dim=1), target) * self.loss_weight 

        return loss

@LOSSES.register_module()
class NoNaNMSE(nn.Module):
    def __init__(self, use_sigmoid=False, loss_weight=None):
        super(NoNaNMSE, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, output, target, ignore_index=-100):
        diff = torch.squeeze(output) - target
        not_nan = ~torch.isnan(diff)
        loss = torch.mean(diff.masked_select(not_nan) ** 2) * self.loss_weight
        return loss
