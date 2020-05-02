#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Object detection module - object.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir

## To-Do List:
# TODO: refactor to handle test and/or valid sets + handle cross validation
"""
import re
import multiprocessing
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Callable, List, Iterable

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.distributed as dist
from ignite.metrics import Accuracy
from torch.utils.data import DataLoader, Dataset
from kedro.pipeline import Pipeline, node

import deepcv.meta as meta
import deepcv.utils as utils
from tests.tests_utils import test_module

__all__ = ['ObjectDetector', 'get_object_detector_pipelines', 'create_model', 'train']
__author__ = 'Paul-Emmanuel Sotir'


class ObjectDetector(meta.base_module.DeepcvModule):
    HP_DEFAULTS = {'architecture': ..., 'act_fn': nn.ReLU, 'batch_norm': None, 'dropout_prob': 0.}

    def __init__(self, input_shape: torch.Size, hp: meta.hyperparams.Hyperparameters):
        super(self.__class__).__init__(self, input_shape, hp)
        self._define_nn_architecture(self._hp['architecture'])
        self._initialize_parameters(self._hp['act_fn'])


def get_object_detector_pipelines():
    p1 = Pipeline([node(meta.hyperparams.merge_hyperparameters, name='merge_hyperparameters', inputs=['params:object_detector', 'params:cifar10_preprocessing'], outputs=['hp']),
                   node(meta.data.preprocess.preprocess, name='preprocess_cifar_dataset', inputs=['trainset', 'testset', 'hp'], outputs=['datasets']),
                   node(create_model, name='create_object_detection_model', inputs=['datasets', 'hp'], outputs=['model']),
                   node(train, name='train_object_detector', inputs=['datasets', 'model', 'hp'], outputs=None)],
                  name='object_detector_training')
    return [p1]


def create_model(datasets: Dict[str, Dataset], hp: meta.hyperparams.Hyperparameters):
    dummy_img, dummy_target = datasets['train_loader'][0][0]
    input_shape = dummy_img.shape
    hp['architecture'][-1]['fully_connected']['out_features'] = np.prod(dummy_target.shape)

    model = ObjectDetector(input_shape, hp['model'])
    return model


def train(datasets: Dict[str, Dataset], model: nn.Module, hp: meta.hyperparams.Hyperparameters):
    # TODO: decide whether we take Datasets or Dataloaders arguments here
    backend_conf = meta.ignite_training.BackendConfig(dist_backend=hp['dist_backend'], dist_url=hp['dist_url'], local_rank=hp['local_rank'])
    metrics = {'accuracy': Accuracy(device='cuda:{backend_conf.local_rank}' if backend_conf.distributed else None)}
    loss = nn.CrossEntropyLoss()
    opt = optim.SGD

    # Create dataloaders from dataset
    if backend_conf.ngpus_current_node > 0 and backend_conf.distributed:
        workers = max(1, (backend_conf.ncpu - 1) // backend_conf.ngpus_current_node)
    else:
        workers = max(1, multiprocessing.cpu_count() - 1)

    max_eval_batch_size = meta.nn.find_best_eval_batch_size(datasets['trainset'][0].shape, model=model, device=backend_conf.device)

    dataloaders = []
    for n, ds in datasets:
        shuffle = True if n == 'trainset' else False
        batch_size = hp['batch_size'] if n == 'trainset' else max_eval_batch_size
        dataloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers))

    return meta.ignite_training.train(hp, model, loss, dataloaders, opt, backend_conf, metrics)


if __name__ == '__main__':
    test_module(__file__)
