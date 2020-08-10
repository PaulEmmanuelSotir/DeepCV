#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Datasets meta module - datasets.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import uuid
import logging
import inspect
import functools
import threading
import collections
from pathlib import Path
from typing import Optional, Type, Union, Iterable, Dict, Any, Tuple, Callable

import PIL
import kedro.io
import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

from deepcv.utils import NL, get_by_identifier, import_tests
from . import training_metadata
from ..types_aliases import *

__all__ = ['TORCHVISION_DATASETS', 'PytorchDataset', 'dataloader_prefetch_batches', 'get_random_subset_dataloader']
__author__ = 'Paul-Emmanuel Sotir'


TORCHVISION_DATASETS = {identifier: value for identifier, value in torchvision.datasets.__dict__.items() if inspect.isclass(value) and issubclass(value, Dataset)}


class PytorchDataset(kedro.io.AbstractDataSet):
    """ Kedro dataset which warps a PyTorch dataset (`torch.utils.data.Dataset`). PyTorch Dataset is only instanciated when `PytorchDataset._load` function is called. """

    def __init__(self, torch_dataset: Union[str, Type[Dataset]], **dataset_kwargs):
        """ Instanciates a `PytorchDataset` Kedro dataset which stores underlying PyTorch dataset (`torch.utils.data.Dataset`) type and arguments needed to instanciate it.
        NOTE: Underlying PyTorch dataset won't be instanciated until `PytorchDataset.load` method is called to avoid unescessary memory usage but given keyword arguments are checked in `PytorchDataset.__init__` to make sure they are valid and can be bound to underlying PyTorch dataset's `__init__` constructor (using `inspect.signature` tooling)
        Args:
            - torch_dataset: PyTorch dataset type. Can either be a Type inheriting from `torch.utils.data.Dataset` or a string identifier which will be parsed by `deepcv.utils.get_by_identifier` in order to retreive specified PyTorch dataset type (It is adviced to use absolute module names when specifying dataset type in a string identifier)
            - dataset_kwargs: Keyword arguments needed to instantiate Pytorch dataset (passed to `__init__` of given PyTorch dataset type).
        """
        super(PytorchDataset, self).__init__()
        self.dataset_kwargs = dataset_kwargs

        if isinstance(torch_dataset, str):
            try:
                # Retreive PyTorch Dataset type from given string identifier
                self.pytorch_dataset = get_by_identifier(torch_dataset)
                if not isinstance(self.pytorch_dataset, Dataset):
                    raise TypeError(f'Error: `{torch_dataset}` should either be a `torch.utils.data.Dataset` or an identifier string'
                                    f' of a type which inherits from `torch.utils.data.Dataset`, got `{self.pytorch_dataset}`')
            except Exception as e:
                msg = f'Error: Dataset warper received a bad argument: ``torch_dataset="{torch_dataset}"`` type cannot be found or instanciated with the following arguments keyword arguments: "{dataset_kwargs}". {NL}Raised exception: "{e}"'
                raise ValueError(msg) from e
        elif issubclass(torch_dataset, Dataset):
            self.pytorch_dataset = torch_dataset
        else:
            raise TypeError(f'Error: `torch_dataset={torch_dataset}` should either be a `torch.utils.data.Dataset` or an identifier string')

        # Make sure given `dataset_kwargs` can be used to instanciate PyTorch dataset (raises TypeError otherwise)
        _bound_args = inspect.signature(self.pytorch_dataset.__init__).bind(**self.dataset_kwargs, self=self.pytorch_dataset)

    def _load(self) -> Dataset:
        """ Instanciate uderlying PyTorch dataset (`torch.utils.data.Dataset`) and return it """
        return self.pytorch_dataset(**self.dataset_kwargs)

    def _save(self): pass

    def _describe(self):
        return {attr: getattr(self, attr) for attr in dir(self)}

    def get_dataset_stats(self) -> deepcv.meta.data.training_metadata.DatasetStats:
        """ Returns various statistics about dataset as a `deepcv.meta.data.training_metadata.DatasetStats` (inherits from `deepcv.meta.data.training_metadata.TrainingMetaData`) """
        # TODO: ...
        raise NotImplementedError


def dataloader_prefetch_batches(dataloader: DataLoader, device: Union[None, str, torch.device] = torch.cuda.current_device()) -> DataLoader:
    """ Monkey-patch `__iter__` method of given `torch.utils.data.DataLoader` in order to prefetch next data batch to device memory during computing/training on previous batch.
    NOTE: In order to data batches being prefetched (to GPU memory), set `pin_memory` to `True` when instanciating DataLoader and provide a `device` which is not 'cpu' nor `None` (e.g. 'gpu').
    NOTE: You won't need to move tensors batches from given dataloader to GPU device memory anymore; i.e., wont need to call `x.to(device)` before computing/training ops.
    Args:
        - dataloader: DataLoader which will be patched in order to prefetch batches (`dataloader.__iter__().__next__` will be monkey-patched)
        - device: Torch device to which data batches are prefetched. If `None` or 'cpu', then batch prefetching is disabled and no changes are made to `dataloader`
    Returns given `dataloader` which will be patched in order to prefetch batches if `device` isn't `None` nor 'cpu' and if `dataloader.pin_memory` is `True` (otherwise returns given DataLoader without any modifications)
    """
    if not dataloader.pin_memory:
        logging.warn(f'Warning: DataLoader wont prefetch data batches: set `pin_memory=True` in your DataLoader when instanciating `{type(dataloader).__name__}`')
    elif device is None or device == 'cpu' or (isinstance(device, torch.device) and device.type == 'cpu'):  # TODO: condition this on 'cuda' instead?
        logging.warn(f'Warning: DataLoader wont prefetch data batches as given `device` argument is `{device}` when prefetching is aimed at GPU(s).')
    else:
        @functools.wraps(dataloader.__iter__)
        def __iter__patch(self: DataLoader, *args, **kwargs):
            iterator = self.__iter__(*args, **kwargs)
            iterator._prefetched_batch = iterator.__next__().to(device=self._prefetch_device, non_blocking=True)
            iterator._dataloader = self

            @functools.wraps(iterator.__next__)
            def __next__patch(iterator_self: Iterable) -> Any:
                if isinstance(iterator_self._prefetched_batch, StopIteration):
                    raise iterator_self._prefetched_batch
                else:
                    batch = iterator_self._prefetched_batch
                    try:
                        iterator_self._prefetched_batch = iterator_self.__next__().to(device=iterator_self._dataloader._prefetch_device, non_blocking=True)
                    except StopIteration as e:
                        # Catch `StopIteration` to raise it later (during following call to `__next__`)
                        iterator_self._prefetched_batch = e
                    return batch

            iterator.__next__ = __next__patch
            return iterator

        dataloader._prefetch_device = device
        dataloader.__iter__ = __iter__patch
    return dataloader


def get_random_subset_dataloader(dataset: Dataset, subset_size: Union[float, int], **dataloader_kwargs) -> DataLoader:
    """ Returns a random subset dataloader sampling data from given dataset, without replacement.
    Args:
        - dataset: PyTorch dataset from which random subset dataloader is sampling data.
        - subset_size: Returned dataloader subsets size. If it is a float, then `subset_size` is intepreted as the subset size fraction of dataset size and should be between 0. and 1.; Otherwise, if it is an interger, `subset_size` is directly interpreted as absolute subset size should be between 0 and `len(dataset)`.
        - dataloader_kwargs: Additional dataloader constructor kwargs, like batch_size, num_workers, pin_memory, ... (dataset, shuffle and sampler arguments are already specified by default)
    """
    if isinstance(subset_size, float):
        assert subset_size > 0. and subset_size <= 1., 'ERROR: `subset_size` should be between 0. and 1. if it is a float.'
        subset_size = max(1, min(len(dataset), int(subset_size * len(dataset) + 1)))
    train_indices = torch.from_numpy(np.random.choice(len(dataset), size=(subset_size,), replace=False))
    return DataLoader(dataset, sampler=SubsetRandomSampler(train_indices), shuffle=True, **dataloader_kwargs)


if __name__ == '__main__':
    cli = import_tests().test_module_cli(__file__)
    cli()
