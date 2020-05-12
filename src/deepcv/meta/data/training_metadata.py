#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Training metadata module - training_metadata.py - `DeepCV`__  
Utilities to keep track of training tasks, hyperparameters (and their eventual hyperparameter space), dataset statistics and experiments from MLFlow.
Builds a training meta-dataset and allows a unified treatment and understanding of models, training procedures, datasets and tasks.
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
    - TODO: Read more in depth Google's approach to meta-datasets: https://github.com/google-research/meta-dataset from this paper: https://arxiv.org/abs/1903.03096 and decide whether it could be relevent to use similar abstractions in deepcv.meta.data.training_tracker
"""
import uuid
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence

import torch
import torch.nn as nn

import deepcv.utils
import deepcv.meta


__all__ = ['TrainingMetaData', 'Task', 'Experiment', 'MetaTracker']
__author__ = 'Paul-Emmanuel Sotir'


class TrainingMetaData:
    def __init__(self, existing_uuid: Optional[uuid.UUID] = None):
        self._uuid = uuid.uuid4() if existing_uuid is None else existing_uuid


class Task(TrainingMetaData):
    def __init__(self, train_loss: torch.nn.modules.loss._Loss, dummy_model_input: torch.Tensor, existing_uuid: Optional[uuid.UUID] = None):
        super(Task, self).__init__(existing_uuid)
        self._train_loss = train_loss
        self._dummy_model_input = dummy_model_input


class Experiment(TrainingMetaData):
    def __init__(self, existing_uuid: Optional[uuid.UUID] = None):
        super(Experiment, self).__init__(existing_uuid)


class MetaTracker:
    def __init__(self, metadataset: deepcv.meta.data.datasets.PytorchDatasetWarper):
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

    def store_dataset_stats(self, trainset: deepcv.meta.data.datasets.PytorchDatasetWarper, dataset_name: str = ''):
        """ Store train dataset statistics and name """
        dataset_stats = trainset.get_dataset_stats()
        raise NotImplementedError
        return dataset_stats

    def update_experiments_from_mlflow(self):
        raise NotImplementedError

    def remove_entry(self, entry_id: Union[uuid.UUID, deepcv.meta.data.datasets.DatasetStats, Experiment, Task, deepcv.meta.hyperparams.HyperparameterSpace, deepcv.meta.hyperparams.Hyperparameters]):
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
        elif is_str and entry_type == 'HyperparameterSpace' or entry_type is deepcv.meta.hyperparams.HyperparameterSpace:
            ...
        elif is_str and entry_type == 'Hyperparameters' or entry_type is deepcv.meta.hyperparams.Hyperparameters:
            ...
        elif is_str and entry_type == 'Experiment' or entry_type is Experiment:
            ...
        elif is_str and entry_type == 'DatasetStats' or entry_type is deepcv.meta.data.datasets.DatasetStats:
            ...

    def reset_all(self):
        """ Removes all metadata entries """
        raise NotImplementedError


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
