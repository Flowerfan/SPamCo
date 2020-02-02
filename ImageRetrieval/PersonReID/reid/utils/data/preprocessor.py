from __future__ import absolute_import
import os.path as osp
import numpy as np

from PIL import Image


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        assert len(dataset) > 0
        self.num_col = len(dataset[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        if self.num_col is 4:
            fname, pid, camid, weight = self.dataset[index]
        elif self.num_col is 3:
            fname, pid, camid = self.dataset[index]
            weight = 1
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid, np.float32(weight)
