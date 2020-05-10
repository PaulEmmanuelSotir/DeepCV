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
test_module_cli = utils.import_tests().test_module_cli

__all__ = ['split_dataset', 'preprocess']
__author__ = 'Paul-Emmanuel Sotir'


def split_dataset(params: Union[Dict[str, Any], meta.hyperparams.Hyperparameters], dataset_or_trainset: meta.data.datasets.PytorchDatasetWarper, testset: Optional[meta.data.datasets.PytorchDatasetWarper] = None) -> Dict[str, meta.data.datasets.PytorchDatasetWarper]:
    func_name = utils.get_str_repr(params, __file__)
    params, _ = meta.hyperparams.to_hyperparameters(params, defaults={'validset_ratio': None, 'testset_ratio': None, 'cache': False})
    logging.info(f'{func_name}: Spliting pytorch dataset into a trainset, testset and eventually a validset: `params="{params}"`')

    # Find testset size to sample from `dataset_or_trainset` if needed
    split_lengths = tuple()
    if testset is None:
        if params['testset_ratio'] is None:
            msg = f'Error: {func_name} function either needs an existing `testset` as argument or you must specify a `testset_ratio` in `params` (probably from parameters/preprocessing YAML config)\nProvided dataset spliting parameters: "{params}"'
            logging.error(msg)
            raise ValueError(msg)
        split_lengths += (int(len(dataset_or_trainset) * params['testset_ratio']),)

    # Find validset size to sample from `dataset_or_trainset` if needed
    if params['validset_ratio'] is not None:
        split_lengths += (int(len(dataset_or_trainset) * params['validset_ratio']),)

    # Return dataset as is if testset is already existing and not validset needs to be sampled
    if testset is not None and params['validset_ratio'] is None:
        return {'trainset': dataset_or_trainset, 'testset': testset}

    # Perform sampling/splitting
    trainset_size = len(dataset_or_trainset) - np.sum(split_lengths)
    if trainset_size < 1:
        msg = f'Error in {func_name}: testset and eventual validset size(s) are too large, there is no remaining trainset samples (maybe dataset is too small (`len(dataset_or_trainset)={len(dataset_or_trainset)}`) or there is a mistake in `testset_ratio={testset_ratio}` or `validset_ration={validset_ration}` values, whcih must be between 0. and 1.).'
        logging.error(msg)
        raise RuntimeError(msg)
    trainset, *testset_and_validset = torch.utils.data.random_split(dataset_or_trainset, (trainset_size, *split_lengths))
    if testset is None:
        testset = testset_and_validset[0]
    validset = testset_and_validset[-1] if params['validset_ratio'] is not None else None

    if params['cache']:
        logging.info(f'{func_name}: Saving resulting dataset to disk (`params["cache"] == True`)...')
        raise NotImplementedError  # TODO: save to (data/03_primary/)
    return {'trainset': trainset, 'validset': validset, 'testset': testset} if validset else {'trainset': trainset, 'testset': testset}


def preprocess(params: Union[Dict[str, Any], meta.hyperparams.Hyperparameters], trainset: meta.data.datasets.PytorchDatasetWarper, testset: meta.data.datasets.PytorchDatasetWarper, validset: Optional[meta.data.datasets.PytorchDatasetWarper] = None) -> Dict[str, DataLoader]:
    """ Main preprocessing procedure. Also make data augmentation if any augmentation recipes have been specified in `hp` (from `parameters.yml`)
    # TODO: create dataloader to preprocess/augment data by batches?
    Args:
        - params:
        - trainset:
        - testset:
        - validset:
    Returns a dict which contains preprocessed and/or augmented 'trainset', 'testset' and 'validset' datasets
    """
    logging.info('Starting pytorch dataset preprocessing procedure...')
    params, _ = meta.hyperparams.to_hyperparameters(params, defaults={'preprocessing': ..., 'cache': False, 'augmentation_reciepe': None, 'normalization_stats': None})

    # Preprocess and augment data according to recipes specified in hyperparameters from YAML config (deepcv/conf/base/parameters.yml)
    for dl in (trainset, validset, testset):
        if not len(dl) > 0:
            raise RuntimeError(f'Error: empty dataloader `{dl}` in {utils.get_str_repr(preprocess, __file__)}')

        # Data augmentation
        if 'augmentations' in params:
            logging.info(f'Applying dataset augmentation reciepe ')
            dl = meta.data.augmentation.apply_augmentation_reciepe(params['augmentations'], dl)

        # Preprocess images
        transforms = _get_img_transforms(hp, trainset)
        dl.transforms = transforms  # TODO: add transforms to dataloader or create a new dataloader if dataloader transforms aren't mutable

        # Preprocess targets
        dl.target_transform = _get_target_transforms

    if params['cache']:
        logging.info('`deepcv.meta.data.preprocess.preprocess` function is saving resulting dataset to disk (`params["cache"] == True`)')
        raise NotImplementedError  # TODO: Save preprocessed dataset to disk (data/04_features/)

    logging.info(f'Pytorch Dataset preprocessing procedure done, returning preprocessed/augmented Dataset(s) ({utils.get_str_repr(preprocess, __file__)}).')
    return {'trainset': trainset, 'validset': validset, 'testset': testset} if validset else {'trainset': trainset, 'testset': testset}


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
    cli = test_module_cli(__file__)
    cli()
