from __future__ import absolute_import

from torch import nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, weights=None):
        loss = F.cross_entropy(inputs, targets, reduction='none')
        if weights is None:
            return loss.mean()
        loss = loss * weights
        return loss.mean()

    def update_clusters(self,clusters):
        self.clusters = clusters
