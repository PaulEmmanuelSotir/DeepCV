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

import deepcv.meta.hyperparams
import deepcv.meta.data.datasets
import deepcv.utils


__all__ = ['PreprocessedDataset', 'split_dataset', 'preprocess', 'stateless_transform_factory', 'np_to_tensor', 'tensor_to_np']
__author__ = 'Paul-Emmanuel Sotir'


class PreprocessedDataset(Dataset):
    """ A Simple PyTorch Dataset which applies given inputs/target transforms to underlying pytorch dataset items """

    def __init__(self, underlying_dataset: Dataset, img_transform: Optional[Callable], target_transform: Optional[Callable] = None, augmentation_transform: Optional[Callable] = None):
        self._unerlying_dataset = underlying_dataset
        self._img_transform = img_transform
        self._target_transform = target_transform
        self._augmentation_transform = augmentation_transform

    def __getitem__(self, index):
        data = self._unerlying_dataset.__getitem__(index)
        if isinstance(data, tuple):
            # We assume first entry is image and any other entries are targets
            x, *ys = data
            if self._img_transform is not None:
                x = self._img_transform(x)
            if self._target_transform is not None:
                ys = (self._target_transform(y) for y in ys)
            if self._augmentation_transform is not None:
                raise NotImplementedError
            return (x, *ys)
        else:
            return self._input_transform(data)

    def __len__(self):
        return len(self._unerlying_dataset)

    def __repr__(self):
        return __repr__(self._unerlying_dataset)


tensor_to_pil = torchvision.transforms.ToPILImage
pil_to_tensor = torchvision.transforms.ToTensor


def stateless_transform_factory(fn: Callable) -> Callable[[], Callable]:
    def _get_transform() -> Callable:
        return fn
    return _get_transform


@stateless_transform_factory
def np_to_tensor(numpy_array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(numpy_array)


@stateless_transform_factory
def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy


def normalize_tensor(trainset: DataLoader, normalization_stats: Optional[Union[torch.Tensor, Sequence[Sequence[float]]]] = None, channels: int = 3) -> torchvision.transforms.Normalize:
    """ Returns a normalizing transform for given dataloader. If there are no given normalization stats (mean and std per channels), then these stats will be processed before returning the transform. """
    stats_shape_error_msg = f'Error: normalization stats should be of shape (2, {channels}) i.e. ((mean + std), (channel count))'
    if isinstance(normalization_stats, torch.Tensor):
        assert normalization_stats.shape == torch.Size((2, channels)), stats_shape_error_msg
        mean, std = normalization_stats[0], normalization_stats[1]
    else:
        assert len(normalization_stats) == 2 and all([len(normalization_stats[i]) == channels for i in range(2)]), stats_shape_error_msg
        mean, std = torch.FloatTensor(normalization_stats[0]), torch.FloatTensor(normalization_stats[1])
    return torchvision.transforms.Normalize(mean, std)


def _process_normalization_stats(trainset: Dataset, to_process: Sequence[str], channels: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    if 'normalization_stats' not in to_process:
        return {}

    # Process mean and std per channel dimension across all trainset image batches
    mean, std = torch.zeros((channels,)), torch.zeros((channels,))
    for input_data in trainset:
        img = input_data if isinstance(input_data, torch.Tensor) else input_data[0]  # Assumes image is the only or first data entry
        mean += img.mean(dim=-3) / len(trainset.dataset)
        std += img.std(dim=-3) / len(trainset.dataset)
    return {'normalization_stats': (mean, std)}


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
            raise ValueError(f'Error: {deepcv.utils.get_str_repr(_define_transforms, __file__)} couldn\'t find "{transform_name}" tranform. Available transforms: "{available_transforms}"')
    return torchvision.transforms.Compose(transforms)


AVAILABLE_TRANSFORMS = {'normalize_tensor': normalize_tensor, 'pil_to_tensor': pil_to_tensor, 'np_to_tensor': np_to_tensor, 'tensor_to_np': tensor_to_np}

# Handle stateful transforms data, e.g. for normalization stats (can either be passed as a (hyper)parameter through `params` or processed at runtime on trainset)
STATEFUL_TRANSFORMS = {'normalize_tensor': {'normalization_stats': ...}}
# Transform state processing functions should take 'trainset' and 'to_process' arguments and return a dict of processed data to be provided to their respective tranform
STATEFUL_DATA_PROCESS = {'normalize_tensor': _process_normalization_stats}


def split_dataset(params: Union[Dict[str, Any], deepcv.meta.hyperparams.Hyperparameters], dataset_or_trainset: Dataset, testset: Optional[Dataset] = None) -> Dict[str, Dataset]:
    func_name = deepcv.utils.get_str_repr(split_dataset, __file__)
    params, _ = deepcv.meta.hyperparams.to_hyperparameters(params, defaults={'validset_ratio': None, 'testset_ratio': None, 'cache': False})
    logging.info(f'{func_name}: Spliting pytorch dataset into a trainset, testset and eventually a validset: `params="{params}"`')
    testset_ratio, validset_ratio = params['testset_ratio'], params['validset_ratio']

    # Find testset size to sample from `dataset_or_trainset` if needed
    split_lengths = tuple()
    if testset is None:
        if testset_ratio is None:
            raise ValueError(f'Error: {func_name} function either needs an existing `testset` as argument or you must specify a `testset_ratio` in `params` '
                             f'(probably from parameters/preprocessing YAML config)\nProvided dataset spliting parameters: "{params}"')
        split_lengths += (int(len(dataset_or_trainset) * testset_ratio),)

    # Find validset size to sample from `dataset_or_trainset` if needed
    if validset_ratio is not None:
        split_lengths += (int(len(dataset_or_trainset) * validset_ratio),)
    elif testset is not None:
        # Testset is already existing and no validset needs to be sampled : return dataset as is
        return {'trainset': dataset_or_trainset, 'testset': testset}

    # Perform sampling/splitting
    trainset_size = int(len(dataset_or_trainset) - np.sum(split_lengths))
    if trainset_size < 1:
        raise RuntimeError(f'Error in {func_name}: testset and eventual validset size(s) are too large, there is no remaining trainset samples '
                           f'(maybe dataset is too small (`len(dataset_or_trainset)={len(dataset_or_trainset)}`) or there is a mistake in `testset_ratio={testset_ratio}` or `validset_ratio={validset_ratio}` values, whcih must be between 0. and 1.).')
    trainset, *testset_and_validset = torch.utils.data.random_split(dataset_or_trainset, (trainset_size, *split_lengths))
    if testset is None:
        testset = testset_and_validset[0]
    validset = testset_and_validset[-1] if validset_ratio is not None else None

    if params['cache']:
        logging.info(f'{func_name}: Saving resulting dataset to disk (`params["cache"] == True`)...')
        raise NotImplementedError  # TODO: save to (data/03_primary/)
    return {'trainset': trainset, 'validset': validset, 'testset': testset} if validset else {'trainset': trainset, 'testset': testset}


def preprocess(params: Union[Dict[str, Any], deepcv.meta.hyperparams.Hyperparameters], datasets: Dict[str, Dataset]) -> Dict[str, PreprocessedDataset]:
    """ Main preprocessing procedure. Also make data augmentation if any augmentation recipes have been specified in `params`.
    Preprocess and augment data according to recipes specified in hyperparameters (`params`) from YAML config (see ./conf/base/parameters.yml)
    # TODO: create dataloader to preprocess/augment data by batches?
    Args:
        - params:
        - datasets: Dict of PyTorch datasets (must contain 'trainset' and 'testset' entries and eventually a 'validset' entry)
    Returns a dict which contains preprocessed and/or augmented 'trainset', 'testset' and 'validset' datasets
    """
    logging.info('Starting pytorch dataset preprocessing procedure...')
    params, _ = deepcv.meta.hyperparams.to_hyperparameters(params, defaults={'transforms': ..., 'target_transforms': [], 'cache': False, 'augmentation_reciepe': None})

    # Filter out sateful transforms which are not used for this preprocessing reciepe
    stateful_transforms = copy.deepcopy(STATEFUL_TRANSFORMS)
    stateful_transforms = {transform: data for transform, data in stateful_transforms.items() if transform in params['transforms'] or transform in params['target_transforms']}
    # Try to fill stateful transforms data from `params`
    stateful_transforms = {transf: {name: params[name] if name in params else ... for name, _ in data.items()} for transf, data in stateful_transforms.items()}

    # Process stateful transforms's missing data from trainset (e.g. mean and variance of trainset images)
    for transform, data in stateful_transforms.items():
        # Process transform state from trainset, and only change data which is not provided in `params`
        to_process = [data_name for data_name, value in data.items() if value == ...]
        if len(to_process) > 0:
            processed_state = STATEFUL_DATA_PROCESS[transform](trainset=datasets['trainset'], to_process=to_process)
            stateful_transforms[transform].update({n: processed_state[n] for n in to_process})
            missing_data = [data_name for data_name, value in data.items() if value == ...]
            if len(missing_data) > 0:
                raise RuntimeError(f"""Error: {deepcv.utils.get_str_repr(preprocess, __file__)} function can`t apply `{transform}` transform, its `STATEFUL_DATA_PROCESS` function did not provided
                                        some nescessary data and `params` (hyper)parameters doesn\'t prodide it neither. `{missing_data}` transform data is missing (you may need to provide it
                                        in hyerparameters or there is an issue in transform\'s `STATEFUL_DATA_PROCESS` function)""")

    # Define image preprocessing transforms
    preprocess_transforms = dict(img_transform=_define_transforms(params['transforms'], AVAILABLE_TRANSFORMS, stateful_transforms))

    # Setup target preprocessing transforms
    if params['target_transforms'] is not None and len(params['target_transforms']) > 0:
        preprocess_transforms['target_transform'] = _define_transforms(params['target_transforms'], AVAILABLE_TRANSFORMS, stateful_transforms)

    # Apply data augmentation
    if params['augmentation_reciepe'] is not None:
        logging.info(f'Applying dataset augmentation reciepe ')
        preprocess_transforms['augmentation_transform'] = deepcv.meta.data.augmentation.apply_augmentation_reciepe(dataset=ds, hp=params['augmentation_reciepe'])

    # Replace datasets with `PreprocessedDataset` instances in order to apply perprocesssing transforms to datasets entries (transforms applied on dataset `__getitem__` calls)
    datasets = {n: PreprocessedDataset(ds, **preprocess_transforms) for n, ds in datasets.items()}

    # If needed, cache/save preprocessed/augmened dataset(s) to disk
    if params['cache']:
        logging.info('`deepcv.meta.data.preprocess.preprocess` function is saving resulting dataset to disk (`params["cache"] == True`)')
        raise NotImplementedError  # TODO: Save preprocessed dataset to disk (data/04_features/)

    logging.info(f'Pytorch Dataset preprocessing procedure ({deepcv.utils.get_str_repr(preprocess, __file__)}) done, returning preprocessed/augmented Dataset(s).')
    return datasets


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
