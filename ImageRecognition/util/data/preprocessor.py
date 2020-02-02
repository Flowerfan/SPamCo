from __future__ import absolute_import
import os.path as osp
import numpy as np

from PIL import Image


class Preprocessor(object):
    def __init__(self, dataset, transform=None):
        super(Preprocessor, self).__init__()
        assert len(dataset) >= 2
        self.data = dataset[0]
        self.targets = dataset[1]
        if len(dataset) > 2:
            assert len(dataset[2]) == len(dataset[0])
            self.weights= dataset[2]
        else:
            self.weights = np.ones(len(dataset[0]), dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        weight = self.weights[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        #  if self.target_transform is not None:
            #  target = self.target_transform(target)

        return img, target, weight
