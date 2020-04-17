#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Preprocessing meta module - preprocess.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Optional, Dict, Tuple, List, Iterable, Union

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .datasets import PytorchDatasetWarper
from ....tests.tests_utils import test_module

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'


def get_normalize_transform(trainset: DataLoader, normalization_stats: Optional[Union[torch.Tensor, List[List[float]]]], channels: int = 3):
    if normalization_stats is None:
        # Process mean and std per channel dimension across all trainset image batches
        mean, std = torch.zeros((channels,)), torch.zeros((channels,))
        for batch in trainset:
            img = batch if isinstance(batch, torch.Tensor) else batch[0]
            mean += img.mean(dim=-3).sum(dim=0) / len(trainset.dataset)
            std += img.std(dim=-3).sum(dim=0) / len(trainset.dataset)
    elif not isinstance(normalization_stats, torch.Tensor):
        assert len(normalization_stats) == 2 and all([len(normalization_stats[i]) == channels for i in range(2)])
        mean, std = torch.FloatTensor(normalization_stats[0]), torch.FloatTensor(normalization_stats[1])
    return torchvision.transforms.Normalize(mean, std)


def preprocess_cifar(trainset: PytorchDatasetWarper, validset: PytorchDatasetWarper, testset: Optional[PytorchDatasetWarper] = None, hp: Dict[str, Any] = {}, augmentation: bool = False) -> Dict[str, Dataset]:
    # Data augmentation
    # TODO: create preprocessing dataloaders, follow augmentation recipes and preprocessing transforms specified in parameters.yml
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
