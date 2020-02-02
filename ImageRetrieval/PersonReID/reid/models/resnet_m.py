from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


__all__ = ['ResNetM', 'resnet18m', 'resnet34m', 'resnet50m', 'resnet101m',
           'resnet152m']


class ResNetM(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0.5, num_classes=0):
        super(ResNetM, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNetM.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNetM.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight,a=0, mode='fan_out')
                init.constant(self.feat.bias, 0)
                # init.constant(self.feat_bn.weight, 1)
                init.normal(self.feat_bn.weight, mean=1, std=0.02)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.leaky_relu(x, 0.1)
        if self.dropout > 0 and self.training:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18m(**kwargs):
    return ResNetM(18, **kwargs)


def resnet34m(**kwargs):
    return ResNetM(34, **kwargs)


def resnet50m(**kwargs):
    return ResNetM(50, **kwargs)


def resnet101m(**kwargs):
    return ResNetM(101, **kwargs)


def resnet152m(**kwargs):
    return ResNetM(152, **kwargs)
