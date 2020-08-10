#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Training metadata module - training_metadata.py - `DeepCV`__  
Utilities to keep track of training tasks, hyperparameters (and their eventual hyperparameter space), dataset statistics and experiments from MLFlow.
Builds a training meta-dataset and allows a unified treatment and understanding of models, training procedures, datasets and tasks.
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
    - TODO: Read more in depth Google's approach to meta-datasets: https://github.com/google-research/meta-dataset from this paper: https://arxiv.org/abs/1903.03096 and decide whether it could be relevent to use similar abstractions in deepcv.meta.data.training_tracker
"""
import abc
import uuid
import types
import collections.abc
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence, List

import torch
import torch.nn as nn

import deepcv.utils
from ..types_aliases import *

__all__ = ['TrainingMetaData', 'DatasetStats', 'Task', 'Experiment', 'HyperparameterSpace', 'Hyperparameters', 'MetaTracker']
__author__ = 'Paul-Emmanuel Sotir'


class TrainingMetaData(abc.ABC):
    def __init__(self, existing_uuid: uuid.UUID = None):
        self._uuid = uuid.uuid4() if existing_uuid is None else existing_uuid


class DatasetStats(TrainingMetaData):
    def __init__(self, existing_uuid: uuid.UUID = None):
        super(self.__class__).__init__(self, existing_uuid)
        # TODO: store dataset datas


class Task(TrainingMetaData):
    def __init__(self, train_loss: torch.nn.modules.loss._Loss, dummy_model_input: torch.Tensor, existing_uuid: uuid.UUID = None):
        super(Task, self).__init__(existing_uuid)
        self._train_loss = train_loss
        self._dummy_model_input = dummy_model_input


class Experiment(TrainingMetaData):
    def __init__(self, existing_uuid: uuid.UUID = None):
        super(Experiment, self).__init__(existing_uuid)


class HyperparameterSpace(TrainingMetaData):
    def __init__(self, existing_uuid: uuid.UUID = None):
        super(HyperparameterSpace, self).__init__(existing_uuid)
        # TODO: implement

    def get_hp_space_overlap(self, hp_space_2: 'HyperparameterSpace'):
        raise NotImplementedError
        overlap = ...
        return overlap


class Hyperparameters(TrainingMetaData, collections.abc.Mapping):
    """ Hyperparameter frozen dict
    Part of this code from [this StackOverflow thread](https://stackoverflow.com/questions/2703599/what-would-a-frozen-dict-be)
    # TODO: refactor deepcv code to make use of this class instead of a simple Dict[str, Any]
    """

    def __init__(self, existing_uuid: uuid.UUID = None, **kwargs):
        TrainingMetaData.__init__(self, existing_uuid)
        collections.abc.Mapping.__init__(self)
        self._store = dict(**kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __getitem__(self, key):
        return self._store[key]

    def __hash__(self):
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash

    def __eq__(self, other: Union[Dict[str, Any], "Hyperparameters"]):
        """ `__eq__` override so that `self.__uuid` isn't taken into account, which makes it consistent with `__hash__` override
        # TODO: May be unescessary as colletions.abc.Mapping defines equality by using 'self.items()': Test it before removing this code...
        """
        if isinstance(other, Hyperparameters):
            return self._store == other._store
        elif isinstance(other, collections.abc.Mapping):
            return self._store == dict(other)
        else:
            # Delegate comparison to the other instance's __eq__.
            raise NotImplemented

    def __ne__(self, other: Any):
        return not self == other

    def get_dict_view(self) -> types.MappingProxyType:
        return types.MappingProxyType(self._store)

    def with_defaults(self, defaults: Union[Dict[str, Any], 'Hyperparameters'], drop_keys_not_in_defaults: bool = False) -> Tuple['Hyperparameters', List[str]]:
        """ Returns a new Hyperaparameter (Frozen dict of hyperparams), with specified defaults
        Args:
            - defaults: Defaults to be applied. Contains default hyperprarmeters with their associated values. If you want to specify some required hyperparameters, set their respective values to ellipsis value `...`.
        Returns a copy of current Hyperarameters (`self`) object updated with additional defaults if not already present in `self`, and a `list` of any missing required hyperparameters names
        """
        defaults = dict(defaults)  # Ensure `defaults` is a dict (may be a generator or another Mapping type)
        new_store = {n: v for n, v in self._store.items() if n in defaults} if drop_keys_not_in_defaults else self._store.copy()
        new_store.update({n: v for n, v in defaults.items() if n not in new_store and v != ...})
        missing_hyperparams = [n for n in defaults if n not in new_store]
        return Hyperparameters(**new_store), missing_hyperparams


class MetaTracker:
    def __init__(self, metadataset):
        self._metadataset = metadataset

    def store_hps(self, hp: Dict[str, Any]):
        raise NotImplementedError

    def store_task(self, train_loss: torch.nn.modules.loss._Loss, dummy_model_input: torch.Tensor, dummy_target: Optional[torch.Tensor]):
        """ Keep track of given task. A tasks is identified by the input data and loss's derivative with respect with target from a dataset.
        If there is no target data, then task is considered to be unsupervised and only identified by its input data.
        Returns stored deepcv.meta.data.training_metadata.Task object
        Args:
            - train_loss:
            - dummy_model_input:
            - dummy_target:
        """
        task = ...
        raise NotImplementedError
        return task

    def store_dataset_stats(self, trainset, dataset_name: str = ''):
        """ Store train dataset statistics and name """
        dataset_stats = trainset.get_dataset_stats()
        raise NotImplementedError
        return dataset_stats

    def update_experiments_from_mlflow(self):
        raise NotImplementedError

    def remove_entry(self, entry_id: Union[uuid.UUID, DatasetStats, Experiment, Task, HyperparameterSpace, Hyperparameters]):
        """ Removes metadata entry from metadataset by its UUID """
        raise NotImplementedError

    def reset(self, entry_type: Union[str, Type]):
        """ Removes all entries of specified type from metadataset
        Args:
            - entry_type: Entrie(s) type to be removed from metadataset (must either be entries's python Type of one of the following strings: 'Task', 'HyperparameterSpace', 'Hyperparameters', 'Experiment' or 'DatasetStats')
        """
        is_str = isinstance(entry_type, str)
        if is_str and entry_type == 'Task' or entry_type is Task:
            ...
        elif is_str and entry_type == 'HyperparameterSpace' or entry_type is HyperparameterSpace:
            ...
        elif is_str and entry_type == 'Hyperparameters' or entry_type is Hyperparameters:
            ...
        elif is_str and entry_type == 'Experiment' or entry_type is Experiment:
            ...
        elif is_str and entry_type == 'DatasetStats' or entry_type is DatasetStats:
            ...

    def reset_all(self):
        """ Removes all metadata entries """
        raise NotImplementedError


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
