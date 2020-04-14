#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Preprocessing meta module - preprocess.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .datasets import PytorchDatasetWarper
from ....tests.tests_utils import test_module

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'


def preprocess_cifar(trainset: PytorchDatasetWarper, validset: PytorchDatasetWarper, testset: Optional[PytorchDatasetWarper] = None, hp: Dict[str, Any] = {}, augmentation: bool = False) -> Dict[str, Dataset]:
    # Data augmentation
    for ds in (trainset.pytorch_dataset, validset.pytorch_dataset, testset.pytorch_dataset):
        if ds is not None and len(ds) > 0:
            if augmentation:
                ds.data, ds.targets = map(_image_augmentation, zip(ds.data, ds.targets))

            # Preprocess images
            ds.transform = _img_preprocess

            # Preprocess targets
            def _target_transform(target: torch.Tensor) -> torch.Tensor:
                return target
            ds.target_transform = _target_transform

    return {'trainset': trainset.pytorch_dataset, 'validset': validset.pytorch_dataset, 'testset': testset.pytorch_dataset}


def _img_preprocess(image: torch.Tensor) -> torch.Tensor:
    return image


def _image_augmentation(image: torch.Tensor, target: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    raise NotImplementedError
    images = [image]
    targets = [target]
    return images, targets


if __name__ == '__main__':
    test_module(__file__)
