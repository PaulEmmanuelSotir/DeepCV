#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Preprocessing meta module - preprocess.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import copy
import logging
import functools
from pathlib import Path
from joblib import Memory
from typing import Optional, Dict, Tuple, List, Iterable, Union, Callable, Any, Sequence

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import PIL
import numpy as np
import albumentations

import deepcv.utils
import deepcv.meta.hyperparams
import deepcv.meta.data.datasets


__all__ = ['PreprocessedDataset', 'fn_to_transform', 'split_dataset', 'preprocess', 'tensor_to_np']
__author__ = 'Paul-Emmanuel Sotir'

# Joblib memory used by `_process_normalization_stats`
joblib_cache_path = Path('./data/03_primary/joblib_cache')
joblib_cache_path.mkdir(parents=True, exist_ok=True)
memory = Memory(joblib_cache_path, verbose=0)


class PreprocessedDataset(Dataset):
    """ A Simple PyTorch Dataset which applies given inputs/target transforms to underlying pytorch dataset items """

    def __init__(self, underlying_dataset: Dataset, img_transform: Optional[Callable], target_transform: Optional[Callable] = None, augmentation_transform: Optional[Callable] = None):
        self._underlying_dataset = underlying_dataset
        self._img_transform = img_transform
        self._target_transform = target_transform
        self._augmentation_transform = augmentation_transform

    def __getitem__(self, index):
        data = self._underlying_dataset.__getitem__(index)
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
            return self._img_transform(data)

    def __len__(self):
        return len(self._underlying_dataset)

    def __repr__(self):
        return f'{PreprocessedDataset.__name__}[{repr(vars(self))}]'


def fn_to_transform(fn: Callable, *transform_args: Iterable[str]) -> Callable[[], Callable]:
    def _get_transform(**transform_kwargs) -> Callable:
        if transform_kwargs:
            # issuperset is used in case there are some arguments in `transform_args` which are optional/defaulted (may raise later if there are missing required arguments).
            if set(transform_args).issuperset(transform_kwargs.keys()):
                raise ValueError(f'Error: `{fn}` transform expected following arguments: `{transform_args}` but got: `{transform_kwargs}`')
            return functools.partial(fn, **transform_kwargs)
        else:
            return fn
    return _get_transform


@fn_to_transform
def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy


# """ Dict listing all available transforms which can be specified with a string identifier.
# NOTE: If other transforms are needed, you can still provide it in YAML config by using its type, for example, `!py!torchvision.datasets.CIFAR100`
# # TODO: Test it (see whether if torchvision transform names are valid)
# """
# AVAILABLE_TRANSFORMS = {'tensor_to_np': tensor_to_np} + {n: v for n, v in vars(torchvision.transforms.transforms).items()}


""" `TRANSFORM_ARGS_PROCESSORS` Maps transforms types with their arguments processing function and an iterable of arguments names which can be processed at runtime by this function.  
If a transform needs arguments which can be processed at runtime instead of beeing provided in YAML config (`parameters.yml`),
you can register its argument(s) processing function in this Dict or decorate argument(s) processing function with `register_transform_processor`.  
NOTE: Registered functions should be a `Callable[['trainset', 'to_process'], Dict[str, Any]]`) which returns a dict of processed arguments to be 
provided to their respective tranform (needed by `deepcv.meta.data.preprocess.preprocess` procedure).
"""
TRANSFORM_ARGS_PROCESSORS = dict()


def register_transform_processor(transform: Union[str, Callable], processable_args_names: Iterable[str]):
    """ Append decorated function to `deepcv.meta.data.preprocess.TRANSFORM_ARGS_PROCESSORS`.  
    If a transform needs arguments which can be proceesed at runtime instead of beeing provided in YAML config (`parameters.yml`),
    you can register its arguments processing function in `deepcv.meta.data.preprocess.TRANSFORM_ARGS_PROCESSORS` Dict or decorate
    this function with `register_transform_processor` (needed by `deepcv.meta.data.preprocess.preprocess` procedure).  
    Args:
        - transform: Transform which needs keyword arguments proceesed by decorated function
        - processable_args_names: Transform arguments names which can be processed by decorated function to be provided to transform constructor
    """
    def _warp(process_fn: Callable[['trainset', 'to_process'], Dict[str, Any]]):
        if transform in TRANSFORM_ARGS_PROCESSORS:
            raise RuntimeError(f'Error: {transform} is already registered in `deepcv.meta.data.preprocess.TRANSFORM_ARGS_PROCESSORS`')
        TRANSFORM_ARGS_PROCESSORS[transform] = (process_fn, processable_args_names)
        return process_fn
    return _warp


@memory.cache
@register_transform_processor(transform=torchvision.transforms.Normalize, processable_args_names=['mean', 'std'])
def _process_normalization_stats(trainset: Dataset, to_process: Sequence[str]) -> Dict[str, torch.Tensor]:
    assert {'mean', 'std'}.issuperset(to_process), f'Error: {deepcv.utils.get_str_repr(_process_normalization_stats)} can only process `mean` or `std`, not: `{to_process}`'

    # Determine channel dimension size (we assume channel dim is the first tensor dim as there isn't batches yet (Dataset))
    dummy_image = trainset[0][0] if isinstance(trainset[0], Tuple) else trainset[0]
    dummy_image = dummy_image if isinstance(dummy_image, torch.Tensor) else torchvision.transforms.ToTensor()(dummy_image)
    channels = dummy_image.shape[0]

    # Process mean and std per channel dimension across all trainset image batches
    stats = {n: torch.zeros((channels,)) for n in to_process}
    for input_data in trainset:
        img = input_data[0] if isinstance(input_data, Tuple) else input_data  # Assumes image is the only or first data entry

        if not isinstance(img, torch.Tensor):
            # Convert images to Tensors if needed
            img = torchvision.transforms.ToTensor()(img)

        dims = tuple(range(1, len(img.shape)))
        if 'mean' in stats:
            stats['mean'] += img.mean(dim=dims) / len(trainset)
        if 'std' in stats:
            stats['std'] += img.std(dim=dims) / len(trainset)

    return stats


def _parse_transforms_specification(transform_identifiers: Sequence, trainset: Dataset, transform_args_processors: Dict = TRANSFORM_ARGS_PROCESSORS) -> torchvision.transforms.Compose:
    """ Parses a transforms specification sequence.  
    Finds transforms type if string identifier is provided, find its arguments from `params`/YAML and/or process it (`TRANSFORM_ARGS_PROCESSORS`) if any, instanciate those transforms and returns their composition.  
    Args:
        - transform_identifiers: Transforms specification to be parsed (probably comes from YAML configuration file, see preprocessing procedures transforms lists in `./conf/base/parameters.yml`)
        - trainset: Trainset Dataset to be provided to functions registered in `transform_args_processors`. See `deepcv.meta.data.preprocess.register_transform_processor` for more details.
        - transform_args_processors: Dict providing function which can process transforms arguments at runtime from `trainset`. See `deepcv.meta.data.preprocess.register_transform_processor` for more details.
    """
    fn_name = deepcv.utils.get_str_repr(_parse_transforms_specification, __file__)
    transforms = []

    for spec in transform_identifiers:
        transform_kwargs = {}
        if isinstance(spec, Dict):
            if not len(spec.items()) == 1:
                raise ValueError(f'Error: {fn_name}: Invalid transform specification, a transform should be specified by a single transform '
                                 f'type/identifer which can eventually be mapped to a dict of keyword arguments')
            if not isinstance(next(iter(spec.values())), Dict):
                raise ValueError(f'Error: {fn_name}: A value mapped to a transform is expected to be a dict of keyword arguments which will '
                                 f'be provided to transform\'s constructor/function, got: `{spec}`')
            # There are user-provided transform keyword arguments in `params` (from YAML)
            spec, transform_kwargs = next(iter(spec.items()))

        # Check transform specification is a valid string identifier (parsed) or Callable
        elif isinstance(spec, str):
            spec = deepcv.utils.get_by_identifier(spec)  # Try to retreive tranform by its string identifier (raises otherwise)
        elif not isinstance(spec, Callable):
            raise ValueError(f'Error: {fn_name} couldn\'t find `{spec}` tranform, transform specification should either be a string identifier or tranform `Callable` type.')

        # Process any missing transform arguments from trainset (only process arguments which are not already provided in `params`/YAML) (e.g. process mean and variance of trainset images)
        if spec in transform_args_processors:
            process_fn, processable_args_names = transform_args_processors[spec]
            to_process = [arg_name for arg_name in processable_args_names if arg_name not in transform_kwargs]
            if len(to_process) > 0:
                # There are missing transform argument(s) to be processed
                processed_state = process_fn(trainset=trainset, to_process=to_process)
                transform_kwargs.update({n: processed_state[n] for n in to_process})

        # Instanciate/call tranform with arguments from spec (YAML) and/or from its runtime processing function
        transforms.append(spec(**transform_kwargs))

    return torchvision.transforms.Compose(transforms)


def preprocess(params: Union[Dict[str, Any], deepcv.meta.hyperparams.Hyperparameters], datasets: Dict[str, Dataset]) -> Dict[str, PreprocessedDataset]:
    """ Main preprocessing procedure. Also make data augmentation if any augmentation recipes have been specified in `params`.
    Preprocess and augment data according to recipes specified in hyperparameters (`params`) from YAML config (see ./conf/base/parameters.yml)
    # TODO: create dataloader to preprocess/augment data by batches?
    Args:
        - params:
        - datasets: Dict of PyTorch datasets (must contain 'trainset' and 'testset' entries and eventually a 'validset' entry)
    Returns a dict which contains preprocessed and/or augmented 'trainset', 'testset' and 'validset' datasets
    """
    fn_name = deepcv.utils.get_str_repr(preprocess, __file__)
    logging.info(f'Starting pytorch dataset preprocessing procedure... ({fn_name})')
    params, _ = deepcv.meta.hyperparams.to_hyperparameters(params, defaults={'transforms': ..., 'target_transforms': [], 'cache': False, 'augmentation_reciepe': None})

    # Define image preprocessing transforms
    preprocess_transforms = dict(img_transform=_parse_transforms_specification(params['transforms'], trainset=datasets['trainset']))

    # Setup target preprocessing transforms
    if params['target_transforms'] is not None and len(params['target_transforms']) > 0:
        preprocess_transforms['target_transform'] = _parse_transforms_specification(params['target_transforms'], trainset=datasets['trainset'])

    # Apply data augmentation
    if params['augmentation_reciepe'] is not None:
        logging.info(f'Applying dataset augmentation reciepe ')
        # TODO: (WIP) use same transforms parsing procedure for augmentation: _parse_transforms_specification(params['augmentation_reciepe']['tranforms'], trainset=datasets['trainset'])
        preprocess_transforms['augmentation_transform'] = deepcv.meta.data.augmentation.apply_augmentation_reciepe(dataset=ds, hp=params['augmentation_reciepe'])

    # Replace datasets with `PreprocessedDataset` instances in order to apply perprocesssing transforms to datasets entries (transforms applied on dataset `__getitem__` calls)
    datasets = {n: PreprocessedDataset(ds, **preprocess_transforms) for n, ds in datasets.items()}

    # If needed, cache/save preprocessed/augmened dataset(s) to disk
    if params['cache']:
        logging.info('`deepcv.meta.data.preprocess.preprocess` function is saving resulting dataset to disk (`params["cache"] == True`)')
        raise NotImplementedError  # TODO: Save preprocessed dataset to disk (data/04_features/)

    logging.info(f'Pytorch Dataset preprocessing procedure ({deepcv.utils.get_str_repr(preprocess, __file__)}) done, returning preprocessed/augmented Dataset(s).')
    return datasets


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


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
