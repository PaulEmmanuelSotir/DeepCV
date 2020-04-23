#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Preprocessing meta module - preprocess.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import logging
from typing import Optional, Dict, Tuple, List, Iterable, Union, Callable, Any

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import deepcv.meta as meta
import deepcv.utils as utils
from ....tests.tests_utils import test_module

__all__ = ['preprocess']
__author__ = 'Paul-Emmanuel Sotir'


def preprocess(hp: meta.hyperparams.Hyperparameters, trainset: meta.data.datasets.PytorchDatasetWarper, testset: meta.data.datasets.PytorchDatasetWarper, validset: Optional[meta.data.datasets.PytorchDatasetWarper] = None) -> Dict[str, DataLoader]:
    """ Main preprocessing procedure. Also make data augmentation if any augmentation recipes have been specified in `hp` (from `parameters.yml`)
    # TODO: create dataloader to preprocess/augment data by batches?
    Args:
        - hp:
        - trainset:
        - testset:
        - validset:
    Returns a dict which contains preprocessed and/or augmented 'trainset', 'testset' and 'validset' datasets
    """
    logging.info('Starting pytorch dataset preprocessing procedure...')
    if 'validset_ratio' in hp:
        if validset is not None:
            logging.warn('Warning: validset is already provided to preprocessing procedure, ignoring `validset_ratio` parameter in preprocessing parameters.')
        else:
            # Create a validset from trainset
            validset_ratio = hp['validset_ratio']
            validset = ...
            raise NotImplementedError
            # TODO: ...

    hp, missings = hp.with_defaults({})
    if len(missings) > 0:
        logging.error(f'Error: Missing mandatory (hyper)parameter(s) in `hp` argument of {utils.get_str_repr(preprocess, __file__)} procedure: Missing parameters: {missings}')

    # Data augmentation
    # TODO: follow augmentation recipes and preprocessing transforms specified in parameters.yml (hp)
    for dl in (trainset, validset, testset):
        if not len(dl) > 0:
            raise RuntimeError(f'Error: empty dataloader `{dl}` in {utils.get_str_repr(preprocess, __file__)}')

        if 'augmentations' in hp:
            logging.info(f'Applying dataset augmentation reciepe ')
            dl = meta.data.augmentation.apply_augmentation_reciepe(hp['augmentations'], dl)

        # Preprocess images
        transforms = _get_img_transforms(hp, trainset)
        dl.transforms = transforms  # TODO: add transforms to dataloader or create a new dataloader if dataloader transforms aren't mutable

        # Preprocess targets
        dl.target_transform = _get_target_transforms

    logging.info(f'Pytorch Dataset preprocessing procedure done, returning preprocessed/augmented Dataset(s) ({utils.get_str_repr(preprocess, __file__)}).')
    return {'trainset': trainset, 'testset': testset} + ({'validset': validset} if validset is not None else {})


def _get_img_transforms(hp: meta.hyperparams.Hyperparameters, dataloader: DataLoader) -> torchvision.transforms.Compose:
    transforms = []
    # TODO: ...
    return torchvision.transforms.Compose(transforms)


def _get_target_transforms(hp: meta.hyperparams.Hyperparameters, dataloader: DataLoader) -> Callable:
    # TODO: ...
    def _target_transform(target: torch.Tensor) -> torch.Tensor:
        return target
    return _target_transform


def _get_normalize_transform(trainset: DataLoader, normalization_stats: Optional[Union[torch.Tensor, List[List[float]]]], channels: int = 3):
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


if __name__ == '__main__':
    test_module(__file__)
