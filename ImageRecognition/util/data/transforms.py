from __future__ import absolute_import

from .augmentation import *
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
import random
import math
import numpy as np
import torch


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self,
                 probability=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class RandomPolicy(object):
    '''
    Class RandomPolicy for augment data
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, data_name='cifar10'):
      if data_name == 'cifar10':
        self.policies = cifar10_policies()

    def pil_wrap(self, img):
        """Convert the PIL image to RGBA"""

        return img.convert('RGBA')


    def pil_unwrap(self, pil_img, img_shape):
        """Converts the PIL RGBA img to a RGB image."""
        pic_array = np.array(pil_img)
        #  import pdb;pdb.set_trace()
        i1, i2 = np.where(pic_array[:, :, 3] == 0)
        pic_array = pic_array[:,:,:3]
        pic_array[i1, i2] = [0, 0, 0]
        return Image.fromarray(pic_array)

    def __call__(self, img):
        policy_idx = np.random.randint(len(self.policies))
        policy = self.policies[policy_idx]
        img_shape = img.size
        pil_img = self.pil_wrap(img)
        for xform in policy:
            assert len(xform) == 3
            name, probability, level = xform
            xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(
              probability, level, img_shape)
            pil_img = xform_fn(pil_img)
        return self.pil_unwrap(pil_img, img_shape)




