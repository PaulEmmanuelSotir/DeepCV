#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Datasets meta module - datasets.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import uuid
import inspect
import threading
import collections
import functools as fn
from pathlib import Path
from typing import Optional, Type, Union, Iterable, Dict, Any

import PIL
import kedro.io
import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

import deepcv.utils
import deepcv.meta.data.training_metadata


__all__ = ['TORCHVISION_DATASETS', 'PytorchDatasetWarper', 'get_random_subset_dataloader']
__author__ = 'Paul-Emmanuel Sotir'


TORCHVISION_DATASETS = {identifier: value for identifier, value in torchvision.datasets.__dict__.items() if inspect.isclass(value) and issubclass(value, Dataset)}


class PytorchDatasetWarper(kedro.io.AbstractDataSet):
    def __init__(self, torch_dataset: Union[str, Type[Dataset]], **dataset_kwargs):
        super(PytorchDatasetWarper, self).__init__()
        self.dataset_kwargs = dataset_kwargs

        if isinstance(torch_dataset, str):
            try:
                # Retreive PyTorch Dataset type from given string identifier
                self.pytorch_dataset = deepcv.utils.get_by_identifier(torch_dataset)
                if not isinstance(self.pytorch_dataset, Dataset):
                    raise TypeError(f'Error: `{torch_dataset}` should either be a `torch.utils.data.Dataset` or an identifier string'
                                    f' of a type which inherits from `torch.utils.data.Dataset`, got `{self.pytorch_dataset}`')
            except Exception as e:
                msg = f'Error: Dataset warper received a bad argument: ``torch_dataset="{torch_dataset}"`` type cannot be found or instanciated with the following arguments keyword arguments: "{dataset_kwargs}". \nRaised exception: "{e}"'
                raise ValueError(msg) from e
        elif issubclass(torch_dataset, Dataset):
            self.pytorch_dataset = torch_dataset
        else:
            raise TypeError(f'Error: `torch_dataset={torch_dataset}` should either be a `torch.utils.data.Dataset` or an identifier string')

        # Make sure given `dataset_kwargs` can be used to instanciate PyTorch dataset (raises TypeError otherwise)
        _bound_args = inspect.signature(self.pytorch_dataset.__init__).bind(**self.dataset_kwargs, self=self.pytorch_dataset)

    def _load(self) -> Dataset:
        return self.pytorch_dataset(**self.dataset_kwargs)

    def _save(self): pass
    def _describe(self): return vars(self)

    def get_dataset_stats(self) -> deepcv.meta.data.training_metadata.DatasetStats:
        """ Returns various statistics about dataset as a `deepcv.meta.data.training_metadata.DatasetStats` (inherits from `deepcv.meta.data.training_metadata.TrainingMetaData`) """
        # TODO: ...
        raise NotImplementedError


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
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
