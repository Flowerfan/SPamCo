from __future__ import absolute_import
import numpy as np
import warnings
import torchvision


__factory=['cifar10', 'svhn']



def create(name, root, download=True, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'cifar10', 'mnist', 'cifar100'
    root : str
        The path to the dataset directory.
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name == 'cifar10':
        data = {}
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=None)
        #  data['train'] = [trainset.train_data, np.array(trainset.train_labels)] ### pytorch 1.0
        data['train'] = [trainset.data, np.array(trainset.targets)] ### pytorch 1.3
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=None)
        #  data['test'] = [testset.test_data, np.array(testset.test_labels)] ### pytorch 1.0
        data['test'] = [testset.data, np.array(testset.targets)] ### pytorch 1.3
        return  data
    elif name == 'svhn':
        data = {}
        trainset = torchvision.datasets.SVHN(root=root, split='train', download=download, transform=None)
        data['train'] = [trainset.data, np.array(trainset.labels)] ### pytorch 1.0
        testset = torchvision.datasets.SVHN(root=root, split='test', download=download, transform=None)
        data['test'] = [testset.data, np.array(testset.labels)] ### pytorch 1.0
        return  data
    elif name == 'mnist':
        data = {}
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=None)
        data['train'] = [np.array(trainset.train_data), np.array(trainset.train_labels)] ### pytorch 1.0
        testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=None)
        data['test'] = [np.array(testset.test_data), np.array(testset.test_labels)] ### pytorch 1.0
        return  data
    else:
        raise KeyError("Unknown dataset:", name)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
