#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Preprocessing meta module - preprocess.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import copy
import logging
from typing import Optional, Dict, Tuple, List, Iterable, Union, Callable, Any, Sequence

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import PIL
import numpy as np

import deepcv.meta as meta
import deepcv.utils as utils


__all__ = ['split_dataset', 'preprocess']
__author__ = 'Paul-Emmanuel Sotir'


AVAILABLE_TRANSFORMS = {'normalize_tensor': normalize, 'pil_to_tensor': pil_to_tensor, 'np_to_tensor': np_to_tensor, 'tensor_to_np': tensor_to_np}

# Handle stateful transforms data, e.g. for normalization stats (can either be passed as a (hyper)parameter through `params` or processed at runtime on trainset)
STATEFUL_TRANSFORMS = {'normalize_tensor': {'normalization_stats': ...}}
# Transform state processing functions should take 'trainset' and 'to_process' arguments and return a dict of processed data to be provided to their respective tranform
STATEFUL_DATA_PROCESS = {'normalize_tensor': _process_normalization_stats}


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
    params, _ = meta.hyperparams.to_hyperparameters(params, defaults={'transforms': ..., 'target_transforms': [], 'cache': False, 'augmentation_reciepe': None})

    # Filter out sateful transforms wich are not used for this preprocessing reciepe
    stateful_transforms = copy.deepcopy(STATEFUL_TRANSFORMS)
    stateful_transforms = {transform: data for transform, data in stateful_transforms if transform in params['transforms'] or transform in params['target_transforms']}
    # Try to fill stateful transforms data from `params`
    stateful_transforms = {transf: {name: params[name] if name in params else ... for name, _ in data} for transf, data in stateful_transforms}

    # Preprocess and augment data according to recipes specified in hyperparameters from YAML config (deepcv/conf/base/parameters.yml)
    for ds_idx, ds in enumerate((trainset, validset, testset)):
        if ds.pytorch_dataset is None or not len(ds.pytorch_dataset) > 0:
            raise RuntimeError(f'Error: empty dataset `{ds}` provided in {utils.get_str_repr(preprocess, __file__)}')

        # Data augmentation
        if params['augmentation_reciepe'] is not None:
            logging.info(f'Applying dataset augmentation reciepe ')
            ds.pytorch_dataset = meta.data.augmentation.apply_augmentation_reciepe(params['augmentation_reciepe'], ds.pytorch_dataset)

        if ds_idx == 0:
            # If we are preprocessing trainset, we process stateful transforms's missing data
            for transform, data in stateful_transforms:
                # Process transform state from trainset, and only change data which is not provided in `params`
                to_process = [data_name for data_name, value in data if value == ...]
                if len(to_process) > 0:
                    processed_state = STATEFUL_DATA_PROCESS[transform](trainset=ds, to_process=to_process)
                    stateful_transforms[transform].update({n: processed_state[n] for n in to_process})
                    missing_data = [data_name for data_name, value in data if value == ...]
                    if len(missing_data) > 0:
                        raise RuntimeError(f"""Error: {utils.get_str_repr(preprocess, __file__)} function can`t apply `{transform}` transform, its `STATEFUL_DATA_PROCESS` function did not provided
                                               some nescessary data and `params` (hyper)parameters doesn\'t prodide it neither. `{missing_data}` transform data is missing (you may need to provide it
                                               in hyerparameters or there is an issue in transform\'s `STATEFUL_DATA_PROCESS` function)""")

        # Define image preprocessing transforms
        ds.pytorch_dataset.transform = _define_transforms(transform_identifiers=params['transforms'], available_transforms=AVAILABLE_TRANSFORMS)

        # Setup target preprocessing transforms
        if params['target_transforms'] is not None and len(params['target_transforms']) > 0:
            ds.pytorch_dataset.target_transform = _define_transforms(transform_identifiers=params['target_transforms'], available_transforms=AVAILABLE_TRANSFORMS)

    # If needed, cache/save preprocessed/augmened dataset(s) to disk
    if params['cache']:
        logging.info('`deepcv.meta.data.preprocess.preprocess` function is saving resulting dataset to disk (`params["cache"] == True`)')
        raise NotImplementedError  # TODO: Save preprocessed dataset to disk (data/04_features/)

    logging.info(f'Pytorch Dataset preprocessing procedure ({utils.get_str_repr(preprocess, __file__)}) done, returning preprocessed/augmented Dataset(s).')
    return {'trainset': trainset, 'validset': validset, 'testset': testset} if validset else {'trainset': trainset, 'testset': testset}


def _define_transforms(transform_identifiers: Sequence[object], available_transforms: Dict[str, Callable], stateful_transforms: Dict[str, Dict[str, Any]] = {}) -> torchvision.transforms.Compose:
        transforms = []
        for transform_name in transform_identifiers:
            if not isinstance(transform_name, str):
                # Assume given transform is already instanciated
                transforms.append(transform_name)
            elif transform_name in available_transforms:
                # Treat stateful transforms as a special case, e.g. 'normalize' transform. (we need to provide data/state to these transform, e.g., mean and variance statistics over dataset for 'normalize' transform)
                transform_factory_kwargs = stateful_transforms[transform_name] if transform_name in stateful_transforms.keys() else {}
                # Instanciate transform
                transforms.append(available_transforms[transform_name](**transform_factory_kwargs))
            else:
                # TODO: parse identifiers like for submodules in 'deepcv.meta.base_module.DeepcvModule' model base class (especially for transforms from torchvision.transforms)
                raise ValueError(f'Error: {utils.get_str_repr(_define_transforms, __file__)} couldn\'t find "{transform_name}" tranform. Available transforms: "{available_transforms}"')
    return torchvision.transforms.Compose(transforms)

def _process_normalization_stats(trainset: Dataset, to_process: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    if 'normalization_stats' not in to_process:
        return {}

    # Process mean and std per channel dimension across all trainset image batches
    mean, std = torch.zeros((channels,)), torch.zeros((channels,))
    for input_data in trainset:
        img = input_data if isinstance(input_data, torch.Tensor) else input_data[0]  # Assumes image is the only or first data entry
        mean += img.mean(dim=-3) / len(trainset.dataset)
        std += img.std(dim=-3) / len(trainset.dataset)
    return {'normalization_stats': (mean, std)}


################################## CUSTOM TRANSFORMS ##################################

def stateless_transform_factory(fn: Callable) -> Callable[[]Callable]:
    def _get_transform() -> Callable:
        return fn
    return _get_transform


tensor_to_pil = torchvision.transforms.ToPILImage
pil_to_tensor = torchvision.transforms.ToTensor

@stateless_transform_factory
def np_to_tensor(numpy_array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(numpy_array)

@stateless_transform_factory
def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy

def normalize_tensor(trainset: DataLoader, normalization_stats: Optional[Union[torch.Tensor, Sequence[Sequence[float]]]] = None, channels: int = 3) -> torchvision.transforms.Normalize:
    """ Returns a normalizing transform for given dataloader. If there are no given normalization stats (mean and std per channels), then these stats will be processed before returning the transform. """
    stats_shape_error_msg = f'Error: normalization stats should be of shape (2, {channels}) i.e. ((mean + std), (channel count))'
    elif issubclass(normalization_stats, torch.Tensor):
        assert normalization_stats.shape == torch.Size((2, channels)), stats_shape_error_msg
        mean, std = normalization_stats[0], normalization_stats[1]
    else:
        assert len(normalization_stats) == 2 and all([len(normalization_stats[i]) == channels for i in range(2)]), stats_shape_error_msg
        mean, std = torch.FloatTensor(normalization_stats[0]), torch.FloatTensor(normalization_stats[1])
    return torchvision.transforms.Normalize(mean, std)


if __name__ == '__main__':
    cli = utils.import_tests().test_module_cli(__file__)
    cli()
