#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Data augmentation meta module - augmentation.py - `DeepCV`__
Some of this python module code is a modified version of [official AugMix implementation](https://github.com/google-research/augmix), under [Apache License 2.0 License](https://github.com/google-research/augmix/blob/master/LICENSE).
# TODO: parse YAML parameters for augmentations reciepes
# TODO: implement various augmentation operators: sharpness, crop, brightness, contrast, tweak_colors, gamma, noise, rotate, translate, scale, smooth_non_linear_deformation
# TODO: implement augmentation based on distilled SinGAN model
# TODO: AugMix augmentation recipe implementation? (see https://arxiv.org/pdf/1912.02781.pdf and parameters.yml)
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Union, Tuple, Callable

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

import torch
import torchvision
import torch.nn as nn

import deepcv.meta.hyperparams as hyperparams
from ....tests.tests_utils import test_module

__all__ = ['AugMixDataset', 'augment_and_mix', 'apply_augmentation']
__author__ = 'Paul-Emmanuel Sotir'

augmentations = [autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y, color, contrast, brightness, sharpness]


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd

    def __getitem__(self, i) -> torch.Tensor:
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), augment_and_mix(x, self.preprocess), augment_and_mix(x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)


def augment_and_mix(image: Image, mixture_width: int = 3, mixture_depth: Union[int, Tuple[int, int]] = [1, 3], severity: int = 3, alpha: float = 1.) -> torch.Tensor:
    """Perform AugMix augmentations and compute mixture
    This augmentation procedure should probably be applied before any preprocessing steps like normalization.
    Args:
        image: input image
        severity: Severity of underlying augmentation operators (between 1 to 10).
        width: Width of augmentation chain
        depth: Depth of augmentation chain. Can either be a constant value or a range of values from which depth is sampled uniformly
        alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
        mixed: Augmented and mixed image, converted to torch.Tensor
    # TODO: sample 'severity', width and alpha parameters uniformly in given range (if they turned to be a tuple of int)?
    """
    pil2tensor = torchvision.transforms.ToTensor()
    ws = np.random.dirichlet([alpha] * mixture_width).astype(np.float32)
    m = np.float32(np.random.beta(alpha, alpha))

    image_tensor = pil2tensor(image)
    mix = torch.zeros_like(image_tensor)
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = np.random.randint(*mixture_depth) if isinstance(mixture_depth, tuple) else mixture_depth
        for _ in range(depth):
            op = np.random.choice(augmentations)
            image_aug = op(image_aug, severity)
        # TODO: avoid convertion to torch.Tensor here and find a way to multiply and combine PIL images without cumbersome convertions to np.ndarray to torch.Tensor
        mix += ws[i] * pil2tensor(image_aug)

    return (1 - m) * image_tensor + m * mix


def apply_augmentation(dataloader: torch.utils.data.DataLoader, hp: hyperparams.Hyperparameters):
    pass


def _int_parameter(level, maxval):
    # TODO: remove these dumb and dumber functions (the dumber is _float_parameter)
    return int(level * maxval / 10)


def _float_parameter(level, maxval):
    # TODO: remove these dumb and dumber functions (nope, the dumber is 👆_int_parameter👆!🤷‍♂️)
    return float(level) * maxval / 10.


def _sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = _int_parameter(_sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = _int_parameter(_sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = _int_parameter(_sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = _float_parameter(_sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = _float_parameter(_sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = _int_parameter(_sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = _int_parameter(_sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR)


def color(pil_img, level):
    """ NOTE: operation that overlaps with ImageNet-C's test set """
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img, level):
    """ NOTE: operation that overlaps with ImageNet-C's test set """
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def brightness(pil_img, level):
    """ NOTE: operation that overlaps with ImageNet-C's test set """
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def sharpness(pil_img, level):
    """ NOTE: operation that overlaps with ImageNet-C's test set """
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


if __name__ == '__main__':
    test_module(__file__)
