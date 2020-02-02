from __future__ import absolute_import

from torchvision.transforms import *
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
import random
import math
import numpy as np
import torch


PARAMETER_MAX = 10

def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / PARAMETER_MAX)

class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, pil_img):
        return self.f(pil_img)


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def pil_transformer(self, probability, level, img_shape):
        def return_function(im):
            if random.random() < probability:
                im = self.xform(im, level, img_shape)
            return im

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)


identity = TransformT('identity', lambda pil_img, level, _: pil_img)
flip_lr = TransformT(
  'FlipLR', lambda pil_img, level, _: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
flip_ud = TransformT(
  'FlipUD', lambda pil_img, level, _: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
# pylint:disable=g-long-lambda
auto_contrast = TransformT(
  'AutoContrast', lambda pil_img, level, _: ImageOps.autocontrast(
    pil_img.convert('RGB')).convert('RGBA'))
equalize = TransformT(
  'Equalize', lambda pil_img, level, _: ImageOps.equalize(
    pil_img.convert('RGB')).convert('RGBA'))
invert = TransformT(
  'Invert', lambda pil_img, level, _: ImageOps.invert(pil_img.convert('RGB')).
  convert('RGBA'))
# pylint:enable=g-long-lambda
blur = TransformT(
  'Blur', lambda pil_img, level, _: pil_img.filter(ImageFilter.BLUR))
smooth = TransformT(
  'Smooth', lambda pil_img, level, _: pil_img.filter(ImageFilter.SMOOTH))


def _rotate_impl(pil_img, level, _):
    """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
    degrees = int_parameter(level, 30)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees)


rotate = TransformT('Rotate', _rotate_impl)


def _posterize_impl(pil_img, level, _):
    """Applies PIL Posterize to `pil_img`."""
    level = int_parameter(level, 4)
    return ImageOps.posterize(pil_img.convert('RGB'),
                              4 - level).convert('RGBA')


posterize = TransformT('Posterize', _posterize_impl)


def _shear_x_impl(pil_img, level, img_shape):
    """Applies PIL ShearX to `pil_img`.

  The ShearX operation shears the image along the horizontal axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((img_shape[0], img_shape[1]), Image.AFFINE,
                             (1, level, 0, 0, 1, 0))


shear_x = TransformT('ShearX', _shear_x_impl)


def _shear_y_impl(pil_img, level, img_shape):
    """Applies PIL ShearY to `pil_img`.

  The ShearY operation shears the image along the vertical axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((img_shape[0], img_shape[1]), Image.AFFINE,
                             (1, 0, 0, level, 1, 0))


shear_y = TransformT('ShearY', _shear_y_impl)


def _translate_x_impl(pil_img, level, img_shape):
    """Applies PIL TranslateX to `pil_img`.

  Translate the image in the horizontal direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateX applied to it.
  """
    level = int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((img_shape[0], img_shape[1]), Image.AFFINE,
                             (1, 0, level, 0, 1, 0))


translate_x = TransformT('TranslateX', _translate_x_impl)


def _translate_y_impl(pil_img, level, img_shape):
    """Applies PIL TranslateY to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateY applied to it.
  """
    level = int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((img_shape[0], img_shape[1]), Image.AFFINE,
                             (1, 0, 0, 0, 1, level))


translate_y = TransformT('TranslateY', _translate_y_impl)


def _crop_impl(pil_img, level, img_shape, interpolation=Image.BILINEAR):
    """Applies a crop to `pil_img` with the size depending on the `level`."""
    cropped = pil_img.crop(
      (level, level, img_shape[0] - level, img_shape[1] - level))
    resized = cropped.resize((img_shape[0], img_shape[1]), interpolation)
    return resized


crop_bilinear = TransformT('CropBilinear', _crop_impl)


def _solarize_impl(pil_img, level, _):
    """Applies PIL Solarize to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had Solarize applied to it.
  """
    level = int_parameter(level, 256)
    return ImageOps.solarize(pil_img.convert('RGB'),
                             256 - level).convert('RGBA')


solarize = TransformT('Solarize', _solarize_impl)


def _enhancer_impl(enhancer):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""

    def impl(pil_img, level, _):
        v = float_parameter(level, 1.8) + .1  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))

ALL_TRANSFORMS = [
  flip_lr, flip_ud, auto_contrast, equalize, invert, rotate, posterize,
  crop_bilinear, solarize, color, contrast, brightness, sharpness, shear_x,
  shear_y, translate_x, translate_y, blur, smooth
]

NAME_TO_TRANSFORM = {t.name: t for t in ALL_TRANSFORMS}
TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()


def cifar10_policies():
  """AutoAugment policies found on CIFAR-10."""
  exp0_0 = [[("Invert", 0.1, 7), ("Contrast", 0.2, 6)],
            [("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)],
            [("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)],
            [("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)],
            [("AutoContrast", 0.5, 8), ("Equalize", 0.9, 2)]]
  exp0_1 = [[("Solarize", 0.4, 5), ("AutoContrast", 0.9, 3)],
            [("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)],
            [("AutoContrast", 0.9, 2), ("Solarize", 0.8, 3)],
            [("Equalize", 0.8, 8), ("Invert", 0.1, 3)],
            [("TranslateY", 0.7, 9), ("AutoContrast", 0.9, 1)]]
  exp0_2 = [[("Solarize", 0.4, 5), ("AutoContrast", 0.0, 2)],
            [("TranslateY", 0.7, 9), ("TranslateY", 0.7, 9)],
            [("AutoContrast", 0.9, 0), ("Solarize", 0.4, 3)],
            [("Equalize", 0.7, 5), ("Invert", 0.1, 3)],
            [("TranslateY", 0.7, 9), ("TranslateY", 0.7, 9)]]
  exp0_3 = [[("Solarize", 0.4, 5), ("AutoContrast", 0.9, 1)],
            [("TranslateY", 0.8, 9), ("TranslateY", 0.9, 9)],
            [("AutoContrast", 0.8, 0), ("TranslateY", 0.7, 9)],
            [("TranslateY", 0.2, 7), ("Color", 0.9, 6)],
            [("Equalize", 0.7, 6), ("Color", 0.4, 9)]]
  exp1_0 = [[("ShearY", 0.2, 7), ("Posterize", 0.3, 7)],
            [("Color", 0.4, 3), ("Brightness", 0.6, 7)],
            [("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)],
            [("Equalize", 0.6, 5), ("Equalize", 0.5, 1)],
            [("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)]]
  exp1_1 = [[("Brightness", 0.3, 7), ("AutoContrast", 0.5, 8)],
            [("AutoContrast", 0.9, 4), ("AutoContrast", 0.5, 6)],
            [("Solarize", 0.3, 5), ("Equalize", 0.6, 5)],
            [("TranslateY", 0.2, 4), ("Sharpness", 0.3, 3)],
            [("Brightness", 0.0, 8), ("Color", 0.8, 8)]]
  exp1_2 = [[("Solarize", 0.2, 6), ("Color", 0.8, 6)],
            [("Solarize", 0.2, 6), ("AutoContrast", 0.8, 1)],
            [("Solarize", 0.4, 1), ("Equalize", 0.6, 5)],
            [("Brightness", 0.0, 0), ("Solarize", 0.5, 2)],
            [("AutoContrast", 0.9, 5), ("Brightness", 0.5, 3)]]
  exp1_3 = [[("Contrast", 0.7, 5), ("Brightness", 0.0, 2)],
            [("Solarize", 0.2, 8), ("Solarize", 0.1, 5)],
            [("Contrast", 0.5, 1), ("TranslateY", 0.2, 9)],
            [("AutoContrast", 0.6, 5), ("TranslateY", 0.0, 9)],
            [("AutoContrast", 0.9, 4), ("Equalize", 0.8, 4)]]
  exp1_4 = [[("Brightness", 0.0, 7), ("Equalize", 0.4, 7)],
            [("Solarize", 0.2, 5), ("Equalize", 0.7, 5)],
            [("Equalize", 0.6, 8), ("Color", 0.6, 2)],
            [("Color", 0.3, 7), ("Color", 0.2, 4)],
            [("AutoContrast", 0.5, 2), ("Solarize", 0.7, 2)]]
  exp1_5 = [[("AutoContrast", 0.2, 0), ("Equalize", 0.1, 0)],
            [("ShearY", 0.6, 5), ("Equalize", 0.6, 5)],
            [("Brightness", 0.9, 3), ("AutoContrast", 0.4, 1)],
            [("Equalize", 0.8, 8), ("Equalize", 0.7, 7)],
            [("Equalize", 0.7, 7), ("Solarize", 0.5, 0)]]
  exp1_6 = [[("Equalize", 0.8, 4), ("TranslateY", 0.8, 9)],
            [("TranslateY", 0.8, 9), ("TranslateY", 0.6, 9)],
            [("TranslateY", 0.9, 0), ("TranslateY", 0.5, 9)],
            [("AutoContrast", 0.5, 3), ("Solarize", 0.3, 4)],
            [("Solarize", 0.5, 3), ("Equalize", 0.4, 4)]]
  exp2_0 = [[("Color", 0.7, 7), ("TranslateX", 0.5, 8)],
            [("Equalize", 0.3, 7), ("AutoContrast", 0.4, 8)],
            [("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)],
            [("Brightness", 0.9, 6), ("Color", 0.2, 8)],
            [("Solarize", 0.5, 2), ("Invert", 0.0, 3)]]
  exp2_1 = [[("AutoContrast", 0.1, 5), ("Brightness", 0.0, 0)],
            [("Equalize", 0.7, 7), ("AutoContrast", 0.6, 4)],
            [("Color", 0.1, 8), ("ShearY", 0.2, 3)],
            [("ShearY", 0.4, 2), ("Rotate", 0.7, 0)]]
  exp2_2 = [[("ShearY", 0.1, 3), ("AutoContrast", 0.9, 5)],
            [("Equalize", 0.5, 0), ("Solarize", 0.6, 6)],
            [("AutoContrast", 0.3, 5), ("Rotate", 0.2, 7)],
            [("Equalize", 0.8, 2), ("Invert", 0.4, 0)]]
  exp2_3 = [[("Equalize", 0.9, 5), ("Color", 0.7, 0)],
            [("Equalize", 0.1, 1), ("ShearY", 0.1, 3)],
            [("AutoContrast", 0.7, 3), ("Equalize", 0.7, 0)],
            [("Brightness", 0.5, 1), ("Contrast", 0.1, 7)],
            [("Contrast", 0.1, 4), ("Solarize", 0.6, 5)]]
  exp2_4 = [[("Solarize", 0.2, 3), ("ShearX", 0.0, 0)],
            [("TranslateX", 0.3, 0), ("TranslateX", 0.6, 0)],
            [("Equalize", 0.5, 9), ("TranslateY", 0.6, 7)],
            [("ShearX", 0.1, 0), ("Sharpness", 0.5, 1)],
            [("Equalize", 0.8, 6), ("Invert", 0.3, 6)]]
  exp2_5 = [[("ShearX", 0.4, 4), ("AutoContrast", 0.9, 2)],
            [("ShearX", 0.0, 3), ("Posterize", 0.0, 3)],
            [("Solarize", 0.4, 3), ("Color", 0.2, 4)],
            [("Equalize", 0.1, 4), ("Equalize", 0.7, 6)]]
  exp2_6 = [[("Equalize", 0.3, 8), ("AutoContrast", 0.4, 3)],
            [("Solarize", 0.6, 4), ("AutoContrast", 0.7, 6)],
            [("AutoContrast", 0.2, 9), ("Brightness", 0.4, 8)],
            [("Equalize", 0.1, 0), ("Equalize", 0.0, 6)],
            [("Equalize", 0.8, 4), ("Equalize", 0.0, 4)]]
  exp2_7 = [[("Equalize", 0.5, 5), ("AutoContrast", 0.1, 2)],
            [("Solarize", 0.5, 5), ("AutoContrast", 0.9, 5)],
            [("AutoContrast", 0.6, 1), ("AutoContrast", 0.7, 8)],
            [("Equalize", 0.2, 0), ("AutoContrast", 0.1, 2)],
            [("Equalize", 0.6, 9), ("Equalize", 0.4, 4)]]
  exp0s = exp0_0 + exp0_1 + exp0_2 + exp0_3
  exp1s = exp1_0 + exp1_1 + exp1_2 + exp1_3 + exp1_4 + exp1_5 + exp1_6
  exp2s = exp2_0 + exp2_1 + exp2_2 + exp2_3 + exp2_4 + exp2_5 + exp2_6 + exp2_7
  return exp0s + exp1s + exp2s
