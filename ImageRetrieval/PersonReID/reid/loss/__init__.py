from __future__ import absolute_import

from .oim import oim, OIM, OIMLoss
from .triplet import TripletLoss
from .tri_clu_loss import TripletClusteringLoss
from .soft_cross_entropy_loss import SoftCrossEntropyLoss

__all__ = [
    'oim',
    'OIM',
    'OIMLoss',
    'TripletLoss',
    'TripletClusteringLoss'
    'SoftCrossEntropyLoss'
]
