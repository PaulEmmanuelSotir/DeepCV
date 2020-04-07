#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Preprocessing meta module - preprocess.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ....tests.tests_utils import test_module

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'


import torchvision


def preprocess_cifar(trainset: Dataset, validset: Dataset, testset: Optional[Dataset] = None, hp: Dict[str, Any], augmentation: bool = False) -> Dict[str, Dataset]:
    # Data augmentation
    for ds in (trainset, validset, testset):
        if ds is not None and len(ds) > 0:
            if augmentation:
                ds.data, ds.targets = map(_image_augmentation, zip(ds.data, ds.targets))

            # Preprocess images
            ds.transform = _img_preprocess

            # Preprocess targets
            def _target_transform(target: torch.Tensor) -> torch.Tensor:
                return target
            ds.target_transform = _target_transform

    return {'trainset': trainset, 'validset': validset, 'testset': testset}


def _img_preprocess(image: torch.Tensor) -> torch.Tensor:
    return image


def _image_augmentation(image: torch.Tensor, target: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    raise NotImplementedError
    images = [image]
    targets = [target]
    return images, targets


if __name__ == '__main__':
    test_module(__file__)
