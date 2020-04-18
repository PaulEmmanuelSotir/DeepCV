#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Object detection module - object.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
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
from torch.utils.data import DataLoader
from kedro.pipeline import Pipeline, node

import deepcv.meta as meta
import deepcv.utils as utils
from tests.tests_utils import test_module

__all__ = ['ObjectDetector', 'get_object_detector_pipelines', 'create_model', 'train']
__author__ = 'Paul-Emmanuel Sotir'


# TODO: refactor to handle test and/or valid sets + handle cross validation
# TODO: add support for residual/dense links
# TODO: make this model fully generic and move it to meta.nn (allow module_creators to be extended and/or overriden)


class ObjectDetector(meta.base_module.DeepcvModule):
    HP_DEFAULTS = {'architecture': ..., 'act_fn': nn.ReLU, 'batch_norm': None, 'dropout_prob': 0.}

    def __init__(self, input_shape: torch.Size, submodule_creators: Dict[str, Callable], hp: meta.hyperparams.Hyperparameters):
        super(self.__class__).__init__(self, input_shape, hp)
        self._net = self._define_nn_architecture(self._hp['architecture'], submodule_creators)
        self._initialize_parameters(self._hp['act_fn'])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._net(x)  # Apply whole neural net architecture


def get_object_detector_pipelines():
    p1 = Pipeline([node(meta.hyperparams.merge_hyperparameters, name='merge_hyperparameters', inputs=['params:object_detector', 'params:cifar10_preprocessing'], outputs=['hp']),
                   node(meta.data.preprocess.preprocess, name='preprocess_cifar_dataset', inputs=['trainset', 'testset', 'hp'], outputs=['dataset']),
                   node(create_model, name='create_object_detection_model', inputs=['dataset', 'hp'], outputs=['model']),
                   node(train, name='train_object_detector', inputs=['dataset', 'model', 'hp'], outputs=None)],
                  name='object_detector_training')
    return [p1]


def create_model(dataset: Dict[str, DataLoader], hp: meta.hyperparams.Hyperparameters):
    dummy_img, dummy_target = dataset['train_loader'][0][0]
    input_shape = dummy_img.shape
    hp['architecture'][-1]['fully_connected']['out_features'] = np.prod(dummy_target.shape)

    module_creators = {'avg_pooling': _create_avg_pooling, 'conv2d': _create_conv2d, 'fully_connected': _create_fully_connected, 'flatten': _create_flatten}
    model = ObjectDetector(input_shape, module_creators, hp['model'])
    return model


def train(datasets: Tuple[torch.utils.data.Dataset], model: nn.Module, hp: meta.hyperparams.Hyperparameters):
    backend_conf = meta.ignite_training.BackendConfig(dist_backend=hp['dist_backend'], dist_url=hp['dist_url'], local_rank=hp['local_rank'])
    metrics = {'accuracy': Accuracy(device='cuda:{backend_conf.local_rank}' if backend_conf.distributed else None)}
    loss = nn.CrossEntropyLoss()
    opt = optim.SGD

    # Create dataloaders from dataset
    if backend_conf.ngpus_current_node > 0 and backend_conf.distributed:
        workers = max(1, (backend_conf.ncpu_per_node - 1) // backend_conf.ngpus_current_node)
    else:
        workers = max(1, multiprocessing.cpu_count() - 1)
    dataloaders = (DataLoader(ds, hp['batch_size'], shuffle=True if i == 0 else False, num_workers=workers) for i, ds in enumerate(datasets))

    return meta.ignite_training.train(hp, model, loss, dataloaders, opt, backend_conf, metrics)


def _create_avg_pooling(layer_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: meta.hyperparams.Hyperparameters) -> nn.Module:
    prev_dim = len(prev_shapes[1:])
    if prev_dim >= 4:
        return nn.AvgPool3d(**layer_params)
    elif prev_dim >= 2:
        return nn.AvgPool2d(**layer_params)
    return nn.AvgPool1d(**layer_params)


def _create_conv2d(layer_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: meta.hyperparams.Hyperparameters) -> nn.Module:
    layer_params['in_channels'] = prev_shapes[-1][1]
    layer = meta.nn.conv_layer(layer_params, hp['act_fn'], hp['dropout_prob'], hp['batch_norm'])
    return layer


def _create_fully_connected(layer_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: meta.hyperparams.Hyperparameters) -> nn.Module:
    layer_params['in_features'] = np.prod(prev_shapes[-1][1:])  # We assume here that features/inputs are given in batches
    if 'out_features' not in layer_params:
        # Handle last fully connected layer (no dropout nor batch normalization for this layer)
        layer_params['out_features'] = self._output_size  # TODO: handle output layer elsewhere
        return meta.nn.fc_layer(layer_params)
    return meta.nn.fc_layer(layer_params, hp['act_fn'], hp['dropout_prob'], hp['batch_norm'])


if __name__ == '__main__':
    test_module(__file__)
