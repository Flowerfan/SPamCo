from __future__ import absolute_import

from torch import nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(SoftCrossEntropyLoss, self).__init__()
        self.reduce = reduce

    def forward(self, inputs, targets, weights=None):
        loss = F.cross_entropy(inputs, targets, reduction='none')
        if weights is None:
            return loss.mean()
        loss = loss.mul_(weights).mean()
        return loss

    def update_clusters(self,clusters):
        self.clusters = clusters
