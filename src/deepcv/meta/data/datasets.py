#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Datasets meta module - datasets.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import uuid
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


TORCHVISION_DATASETS = {identifier: value for identifier, value in torchvision.datasets.__dict__.items() if value is Dataset}


class PytorchDatasetWarper(kedro.io.AbstractDataSet):
    def __init__(self, torch_dataset: Union[str, Type[Dataset]], **dataset_kwargs):
        super(PytorchDatasetWarper, self).__init__()
        if isinstance(torch_dataset, str):
            try:
                self.pytorch_dataset = deepcv.utils.get_by_identifier(torch_dataset)(**dataset_kwargs)
            except Exception as e:
                raise ValueError(f'Error: Dataset warper received a bad argument: ``torch_dataset="{torch_dataset}"`` doesn\'t match type identifier criterias.') from e
        else:
            self.pytorch_dataset = torch_dataset(**dataset_kwargs)

    def _load(self): pass
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
