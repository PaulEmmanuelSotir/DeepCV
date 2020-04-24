#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Data augmentation meta module - augmentation.py - `DeepCV`__
Some of this python module code is a modified version of [official AugMix implementation](https://github.com/google-research/augmix), under [Apache License 2.0 License](https://github.com/google-research/augmix/blob/master/LICENSE).
.. See Google Research/DeepMind [ICLR 2020 AugMix paper](https://arxiv.org/pdf/1912.02781.pdf)
# TODO: parse YAML parameters for augmentations reciepes
# TODO: implement various augmentation operators: sharpness, crop, brightness, contrast, tweak_colors, gamma, noise, rotate, translate, scale, smooth_non_linear_deformation
# TODO: implement augmentation based on distilled SinGAN model
# TODO: AugMix augmentation recipe implementation? (see https://arxiv.org/pdf/1912.02781.pdf and parameters.yml)
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Union, Tuple, Callable, Mapping

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

import torch
import torchvision
import torch.nn as nn

import deepcv.meta.hyperparams as hyperparams
from ....tests.tests_utils import test_module

__all__ = ['apply_augmentation_reciepe', 'augment_and_mix', 'autocontrast', 'equalize', 'posterize',
           'rotate', 'solarize', 'shear_x', 'shear_y', 'translate_x', 'translate_y', 'color', 'contrast', 'brightness', 'sharpness']
__author__ = 'Paul-Emmanuel Sotir'

AUGMENTATION_OPS = [autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y, color, contrast, brightness, sharpness]


def apply_augmentation_reciepe(dataloader: torch.utils.data.DataLoader, hp: Union[hyperparams.Hyperparameters, Mapping]):
    """ Applies listed augmentation transforms with given configuration from `hp` Dict.
    .. See [deepcv/conf/base/parameters.yml](./conf/base/parameters.yml) for examples of augmentation reciepe specification
    Args:
        - dataloader: Dataset dataloader on which data augmnetation is performed
        - hp: Augmentation hyperparameters (Mapping or hyperparams.Hyperparameters object), must at least contain `transforms` entry, see `hp.with_defaults({...})` in this function code or [augmentation reciepes spec. in ./conf/base/parameters.yml](./conf/base/parameters.yml)
    Returns a new torch.utils.data.DataLoader which samples data from newly created augmented dataset
    """
    # Parse hyperparameters
    if not isinstance(hp, hyperparams.Hyperparameters):
        hp = hyperparams.Hyperparameters(**hp)
    hp, missing_hyperparams = hp.with_defaults({'transforms': ..., 'keep_same_input_shape': False, 'random_transform_order': True,
                                                'augmentation_ops_depth': [1, 4], 'augmentations_per_image': [0, 3], 'augmix': None})
    if hp.get('augmix'):
        augmix_defaults = {'augmentation_chains_count': ..., 'transform_chains_dirichlet': ..., 'mix_with_original_beta_distr': ...}
        augmix_params = hyperparams.Hyperparameters(hp['augmix']).with_defaults(augmix_defaults)

    transforms = hp['transforms']

    if hp['keep_same_input_shape']:
        # TODO: resize (scale and/or crop?) output image to its original size
    raise NotImplementedError


def augment_and_mix(image: Image, mixture_width: int = 3, mixture_depth: Union[int, Tuple[int, int]] = [1, 3], severity: int = 0.3, alpha: float = 1.) -> torch.Tensor:
    """Perform AugMix augmentations on PIL images and compute mixture
    This augmentation procedure should probably be applied before any preprocessing steps like normalization.
    NOTE: AugMix augmentation reciepe is often used along with JSD (Jensen-Shannon-Divergence, implemented in [deepcv.meta.contrastive module](./src/deepcv/meta/contrastive.py)). JSD is a contrastive loss proposed in [ICLR 2020 AugMix paper](https://arxiv.org/pdf/1912.02781.pdf) which enforces trained model to be invariant to augmentation transforms as long as target remains unchanged.
    Args:
        image: input PIL image to augment
        severity: Severity of underlying augmentation operators (between 0. and 1.).
        width: Width of augmentation chain
        depth: Depth of augmentation chain. Can either be a constant value or a range of values from which depth is sampled uniformly
        alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
        mixed: Augmented and mixed image, converted to torch.Tensor
    # TODO: sample 'severity', width and alpha parameters uniformly in given range (if they turned to be a tuple of int)?
    """
    assert severity >= 0. and severity <= 1., f"Assert failed in {utils.get_str_repr(augment_and_mix, __file__)}: `severity` argument should be a float in [0;1] range."
    if severity == 0.:
        return pil2tensor(image)

    pil2tensor = torchvision.transforms.ToTensor()
    ws = np.random.dirichlet([alpha] * mixture_width).astype(np.float32)
    m = np.float32(np.random.beta(alpha, alpha))

    with torch.no_grad():
        image_tensor = pil2tensor(image)
        mix = torch.zeros_like(image_tensor)
        for i in range(mixture_width):
            image_aug = image.copy()
            depth = np.random.randint(*mixture_depth) if isinstance(mixture_depth, tuple) else mixture_depth
            for _ in range(depth):
                op = np.random.choice(AUGMENTATION_OPS)
                image_aug = op(image_aug, severity)
            # TODO: avoid convertion to torch.Tensor here and find a way to multiply and combine PIL images without cumbersome convertions
            mix += ws[i] * pil2tensor(image_aug)

        return (1 - m) * image_tensor + m * mix


def autocontrast(pil_img: Image, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img: Image, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img: Image, severity: float, max_color_bits: int = 5, min_color_bits: int = 2):
    bits = np.random.uniform(low=min_color_bits, high=max_color_bits)
    return ImageOps.posterize(pil_img, int(bits + (1. - severity) * (8. - bits)))


def rotate(pil_img: Image, severity: float, max_angle: float = 30.):
    degrees = np.random.uniform(low=-severity, high=severity) * max_angle
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img: Image, severity: float, max_threshold: int = 220,  min_threshold: int = 50):
    threshold = int(np.random.uniform(low=(1. - severity) * (max_threshold - min_threshold) + min_threshold, high=max_threshold))
    return ImageOps.solarize(pil_img, threshold)


def shear_x(pil_img: Image, severity: float, max_shear: float = 0.3):
    shear = np.random.uniform(low=-severity, high=severity) * max_shear
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), resample=Image.BILINEAR)


def shear_y(pil_img: Image, severity: float, max_shear: float = 0.3):
    shear = np.random.uniform(low=-severity, high=severity) * max_shear
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0), resample=Image.BILINEAR)


def translate_x(pil_img: Image, severity: float, max_translation: float = 1./3.):
    translation = int(np.random.uniform(low=-severity, high=severity) * (pil_img.size[0] * max_translation))
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, translation, 0, 1, 0), resample=Image.BILINEAR)


def translate_y(pil_img: Image, severity: float, max_translation: float = 1./3.):
    translation = int(np.random.uniform(low=-severity, high=severity) * (pil_img.size[1] * max_translation))
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, translation), resample=Image.BILINEAR)


def color(pil_img: Image, severity: float, max_enhance_factor: float = 1.8, min_enhance_factor: float = 0.3):
    """ NOTE: operation that overlaps with ImageNet-C's test set """
    enhance = np.random.uniform(low=min_enhance_factor, high=max(severity * max_enhance_factor, min_enhance_factor))
    return ImageEnhance.Color(pil_img).enhance(enhance)


def contrast(pil_img: Image, severity: float, max_enhance_factor: float = 1.8, min_enhance_factor: float = 0.3):
    """ NOTE: operation that overlaps with ImageNet-C's test set """
    enhance = np.random.uniform(low=min_enhance_factor, high=max(severity * max_enhance_factor, min_enhance_factor))
    return ImageEnhance.Contrast(pil_img).enhance(enhance)


def brightness(pil_img: Image, severity: float, max_enhance_factor: float = 1.8, min_enhance_factor: float = 0.3):
    """ NOTE: operation that overlaps with ImageNet-C's test set """
    enhance = np.random.uniform(low=min_enhance_factor, high=max(severity * max_enhance_factor, min_enhance_factor))
    return ImageEnhance.Brightness(pil_img).enhance(enhance)


def sharpness(pil_img: Image, severity: float, max_enhance_factor: float = 1.8, min_enhance_factor: float = 0.3):
    """ NOTE: operation that overlaps with ImageNet-C's test set """
    enhance = np.random.uniform(low=min_enhance_factor, high=max(severity * max_enhance_factor, min_enhance_factor))
    return ImageEnhance.Sharpness(pil_img).enhance(enhance)


if __name__ == '__main__':
    test_module(__file__)
