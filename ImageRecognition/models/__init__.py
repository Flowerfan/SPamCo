from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .wrn import *
from .shake_shake import shake_shake
from .shake_drop import *
from .shake_drop2 import shake_drop2
from .mnist_net import mnist_net1,mnist_net2,mnist_net3



__factory = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'vgg': VGG,
    'efficientnet': EfficientNetB0,
    'densenet121': DenseNet121,
    'shufflenet': ShuffleNetG2,
    'mobilenet': MobileNet,
    'mobilenetv2': MobileNetV2,
    'wrn': WideResNet,
    'shake_shake': shake_shake,
    'shake_drop': shake_drop_net,
    'shake_drop2': shake_drop2,
    'mnist_net1': mnist_net1,
    'mnist_net2': mnist_net2,
    'mnist_net3': mnist_net3,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.
    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
