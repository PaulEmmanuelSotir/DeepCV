#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Object detection module - object.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
    - TODO: refactor to handle test and/or valid sets + handle cross validation
"""
import re
import multiprocessing
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Callable, List, Iterable, Union

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

import ignite
import numpy as np
from ignite.metrics import Accuracy
from kedro.pipeline import Pipeline, node

import deepcv.utils
import deepcv.meta.nn
import deepcv.meta.base_module
import deepcv.meta.hyperparams
import deepcv.meta.data.preprocess
import deepcv.meta.ignite_training
from deepcv.meta.data.datasets import BatchPrefetchDataLoader

__all__ = ['ObjectDetector', 'get_object_detector_pipelines', 'create_model', 'train']
__author__ = 'Paul-Emmanuel Sotir'


class ObjectDetector(deepcv.meta.base_module.DeepcvModule):
    HP_DEFAULTS = {'architecture': ..., 'act_fn': nn.ReLU, 'batch_norm': None, 'dropout_prob': 0.}

    def __init__(self, input_shape: torch.Size, hp: Union[deepcv.meta.hyperparams.Hyperparameters, Dict[str, Any]]):
        super().__init__(input_shape, hp)
        self.define_nn_architecture(self._hp['architecture'])
        self.initialize_parameters(self._hp['act_fn'])


def get_object_detector_pipelines() -> Dict[str, Pipeline]:
    p1 = Pipeline([node(deepcv.utils.setup_cudnn, name='setup_cudnn_and_seed', inputs=dict(deterministic='params:object_detector_training.deterministic', seed='params:object_detector_training.seed'), outputs=None),
                   node(deepcv.meta.data.preprocess.split_dataset, name='split_dataset',
                        inputs=dict(params='params:split_dataset', dataset_or_trainset='cifar10_train', testset='cifar10_test'), outputs='datasets'),
                   node(deepcv.meta.data.preprocess.preprocess, name='preprocess', inputs=dict(
                       datasets='datasets', params='params:cifar10_preprocessing'), outputs='preprocessed_datasets'),
                   node(create_model, name='create_object_detection_model', inputs=['preprocessed_datasets', 'params:object_detector_model'], outputs='model'),
                   node(train, name='train_object_detector', inputs=['preprocessed_datasets', 'model', 'params:object_detector_training'], outputs='ignite_state')],
                  tags=['train', 'detection'])
    return {'object_detector_training': p1}


def create_model(datasets: Dict[str, Dataset], model_params: Union[deepcv.meta.hyperparams.Hyperparameters, Dict[str, Any]]):
    # Determine input and output shapes
    dummy_img, dummy_target = datasets['trainset'][0]
    input_shape = dummy_img.shape
    if not hasattr(model_params['architecture'][-1]['fully_connected'], 'out_features'):
        # Architecture's last FC output layer doesn't specify output features size, so we deduce it from dataset's target classes or target shape
        classes = deepcv.utils.recursive_getattr(datasets['trainset'], 'classes', recurse_on_type=Dataset)
        if classes is not None:
            model_params['architecture'][-1]['fully_connected']['out_features'] = len(classes)
        elif isinstance(dummy_target, torch.Tensor):
            model_params['architecture'][-1]['fully_connected']['out_features'] = np.prod(dummy_target.shape)

    # Create ObjectDetector model
    model = ObjectDetector(input_shape, model_params)
    return model


def train(datasets: Dict[str, Dataset], model: nn.Module, hp: Union[deepcv.meta.hyperparams.Hyperparameters, Dict[str, Any]]) -> ignite.engine.State:
    """ Train object detector model
    .. See Fastai blog post about AdamW for more details about optimizer: https://www.fast.ai/2018/07/02/adam-weight-decay/
    """
    backend_conf = deepcv.meta.ignite_training.BackendConfig(**hp['backend_conf'])
    metrics = {'accuracy': Accuracy(device=backend_conf.device if backend_conf.distributed else None)}
    loss = nn.CrossEntropyLoss()
    opt = optim.AdamW

    # Determine num_workers for DataLoaders
    if backend_conf.ngpus_current_node > 0 and backend_conf.distributed:
        workers = max(1, (backend_conf.ncpu - 1) // backend_conf.ngpus_current_node)
    else:
        workers = max(1, multiprocessing.cpu_count() - 1)

    # Create dataloaders from dataset
    dataloaders = []
    for n, ds in datasets.items():
        shuffle = True if n == 'trainset' else False
        batch_size = hp['batch_size'] if n == 'trainset' else hp['batch_size'] * 32
        # TODO: workaround ignite which is incompatible with custom DataLoader classes and make usage of deepcv.meta.data.datasets.BatchPrefetchDataLoader
        dataloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                      pin_memory=not backend_conf.is_cpu))

    return deepcv.meta.ignite_training.train(hp, model, loss, dataloaders, opt, backend_conf, metrics)


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
