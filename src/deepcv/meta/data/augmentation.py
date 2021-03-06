#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Data augmentation meta module - augmentation.py - `DeepCV`__  
Some of this python module code is a modified version of [official AugMix implementation](https://github.com/google-research/augmix), under [Apache License 2.0 License](https://github.com/google-research/augmix/blob/master/LICENSE).  
.. See Google Research/DeepMind [ICLR 2020 AugMix paper](https://arxiv.org/pdf/1912.02781.pdf)  
.. moduleauthor:: Paul-Emmanuel Sotir  

*To-Do List*  
    - TODO: Use albumentation and/or DALI augmentation transforms instead of torchvision's. See DALI augmentation example: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/image_processing/augmentation_gallery.html
        DALI have GPU support but Albumentation still features good performances and, more importntly, have better handling of targets made of image transform-dependent data. Albumentation Image transforms may support Masks, BoundingBoxes and Keypoints targets
    - TODO: finish implementation of YAML parsing of augmentations reciepes specs
    - TODO: implement various augmentation operators: sharpness, crop, brightness, contrast, tweak_colors, gamma, noise, rotate, translate, scale, smooth_non_linear_deformation
    - TODO: implement augmentation based on distilled SinGAN model
    - TODO: AugMix augmentation recipe implementation? (see https://arxiv.org/pdf/1912.02781.pdf and parameters.yml)
    - TODO: make sure Pillow-Simd is well installed and well performing. Otherwise, move to another transform framework
    - TODO: Look into Albumentations Python package/library (or make sure to transform eventual targets to be coherent with transformed images)
"""
import functools
from typing import Union, Tuple, Callable, Mapping, Sequence, Any, Dict

import numpy as np
import albumentations
from PIL import Image, ImageOps, ImageEnhance

import torch
import torch.nn
import torchvision

import deepcv.utils
from .. import hyperparams
from ..types_aliases import *


__all__ = ['apply_augmentation_reciepe', 'augment_and_mix', 'autocontrast', 'equalize', 'posterize',
           'rotate', 'solarize', 'shear_x', 'shear_y', 'translate_x', 'translate_y', 'color', 'contrast', 'brightness', 'sharpness']
__author__ = 'Paul-Emmanuel Sotir'


albumentations.HorizontalFlip, albumentations.IAAPerspective, albumentations.ShiftScaleRotate, albumentations.CLAHE, albumentations.RandomRotate90,
albumentations.Transpose, albumentations.ShiftScaleRotate, albumentations.Blur, albumentations.OpticalDistortion, albumentations.GridDistortion, albumentations.HueSaturationValue,
albumentations.IAAAdditiveGaussianNoise, albumentations.GaussNoise, albumentations.MotionBlur, albumentations.MedianBlur, albumentations.IAAPiecewiseAffine,
albumentations.IAASharpen, albumentations.IAAEmboss, albumentations.RandomBrightnessContrast, albumentations.Flip 

albumentations.OneOf, albumentations.Compose


albumentations.core.composition.BboxParams
albumentations.core.composition.KeypointParams
aug_pipeline = Compose(tranforms, keypoint_params, bbox_params, additional_targets= Dict) # transform, OneOf, OneOrOther
force_apply=True # Ignores p parameter
# TODO: use ReplayCompose in case of deterministic training or only rely on fixed random seed?? 
# Use additional_targets={'image0': 'image', 'image1': 'image'} in case you need to transform multiple images with the exact same pipeline and trasforms
aug_pipeline(image, mask=mask, keypoints=keypoints, bboxes=bboxes)


def apply_transform(transform: str, img: Union[Image, torch.Tensor, np.ndarray], severity: float = 1., transform_params: Dict[str, Any] = None, keypoints_targets = None, mask_targets= None, bounding_box_targets: Sequence[]= None, other_data: Any = None):
    """
    """
    if isinstance(transform, str):
        tranform = ALBUMENTATIONS_TRANSFORMS[transform_name] 
    if targets is not None and len(targets) > 0:
        pass
    return tranform(img, severity, )

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


AUGMENTATION_OPS = [autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y, color, contrast, brightness, sharpness]


def apply_augmentation_reciepe(datasets: Tuple[torch.utils.data.Dataset], params: HYPERPARAMS_T) -> Callable:
    """ Applies listed augmentation transforms with given configuration from `params` Dict.
    .. See [deepcv/conf/base/parameters.yml](./conf/base/parameters.yml) for examples of augmentation reciepe specification
    Args:
        - datasets: Tuple of Datasets on which data augmentation is performed
        - params: Augmentation hyperparameters (Mapping or deepcv.meta.hyperparams.Hyperparameters object), must at least contain `transforms` entry, see `params.with_defaults({...})` in this function code or [augmentation reciepes spec. in ./conf/base/parameters.yml](./conf/base/parameters.yml)
    Returns a transform which returns augmented image(s) from original image
    """
    params, _ = hyperparams.to_hyperparameters(params, {'transforms': ..., 'keep_same_input_shape': False, 'random_transform_order': True,
                                                        'augmentation_ops_depth': [1, 4], 'augmentations_per_image': [0, 3], 'augmix': None})
    transforms = []

    if params.get('augmix') is not None:
        augmix_defaults = {'augmentation_chains_count': ..., 'transform_chains_dirichlet': ..., 'mix_with_original_beta': ...}
        augmix_params, _ = hyperparams.to_hyperparameters(params['augmix'], augmix_defaults)
        augmix_transform = functools.partial(augment_and_mix, chains_depth=params['augmentation_ops_depth'], **augmix_params)
        # TODO: apply augmentation transforms
        raise NotImplementedError
    else:
        raise NotImplementedError
        # TODO: sample randomly augmentation ops
        for t in params['transforms']:
            transforms.append(t)

    if params['keep_same_input_shape']:
        raise NotImplementedError
        # TODO: resize (scale and/or crop?) output image to its original size
    return torchvision.transforms.Compose(transforms)


def augment_and_mix(image: Image, augmentation_chains_count: int = 3, chains_depth: Union[int, Tuple[int, int]] = [1, 3], severity: int = 0.3, transform_chains_dirichlet: float = 1., mix_with_original_beta: float = 1.) -> torch.Tensor:
    """Perform AugMix augmentations on PIL images and compute mixture
    This augmentation procedure should probably be applied before any preprocessing steps like normalization.
    NOTE: AugMix augmentation reciepe is often used along with JSD (Jensen-Shannon-Divergence, implemented in [deepcv.meta.contrastive module](./src/deepcv/meta/contrastive.py)). JSD is a contrastive loss proposed in [ICLR 2020 AugMix paper](https://arxiv.org/pdf/1912.02781.pdf) which enforces trained model to be invariant to augmentation transforms as long as target remains unchanged.
    Args:
        - image: input PIL image to augment
        - severity: Severity of underlying augmentation operators (between 0. and 1.).
        - augmentation_chains_count: Number of augmentation chains (width)
        - chains_depth: Depth of augmentation chain(s). Can either be a constant value or a range of values from which depth is sampled uniformly
        - transform_chains_dirichlet: Probability coefficient Dirichlet distribution used to mix each augmentation chains together.
        - mix_with_original_beta: Probability coefficient for Beta distributions used to mix original image with mixed augmented images.
    Returns:
        mixed: Augmented and mixed image, converted to torch.Tensor
    # TODO: sample 'severity', width and alpha parameters uniformly in given range (if they turned to be a tuple of int)?
    """
    assert severity >= 0. and severity <= 1., f"Assert failed in {deepcv.utils.get_str_repr(augment_and_mix, __file__)}: `severity` argument should be a float in [0;1] range."
    pil2tensor = torchvision.transforms.ToTensor()
    if severity == 0.:
        return pil2tensor(image)

    # TODO: read image with opencv or convert PIL image to np.ndarray before transforming it with albumentation (For more details, see https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/migrating_from_torchvision_to_albumentations.ipynb)

    ws = np.random.dirichlet([transform_chains_dirichlet] * augmentation_chains_count).astype(np.float32)
    m = np.float32(np.random.beta(mix_with_original_beta, mix_with_original_beta))

    with torch.no_grad():
        image_tensor = pil2tensor(image)
        mix = torch.zeros_like(image_tensor)
        for i in range(augmentation_chains_count):
            image_aug = image.copy()
            depth = np.random.randint(*chains_depth) if isinstance(chains_depth, tuple) else chains_depth
            # TODO: apply albumentation transform pipeline here instead of looping over torchvision transforms
            for _ in range(depth):
                op = np.random.choice(AUGMENTATION_OPS)
                image_aug = op(image_aug, severity)
            # TODO: avoid convertion to torch.Tensor here and find a way to multiply and combine PIL images without cumbersome convertions
            mix += ws[i] * pil2tensor(image_aug)

        return (1 - m) * image_tensor + m * mix


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
