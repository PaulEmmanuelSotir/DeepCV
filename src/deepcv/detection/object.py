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
from deepcv.meta.data.datasets import batch_prefetch_dataloader_patch

__all__ = ['get_object_detector_pipelines', 'create_model', 'train']
__author__ = 'Paul-Emmanuel Sotir'


def get_object_detector_pipelines() -> Dict[str, Pipeline]:
    """ Defines all Kedro nodes and pipelines of object detector model: datasets setup/preprocessing, object detector training and object detector hyperparameters search pipelines. """
    preprocess_node = node(deepcv.meta.data.preprocess.preprocess, name='preprocess',
                           inputs=dict(dataset_or_trainset='cifar10_train', testset='cifar10_test', params='params:cifar10_preprocessing'),
                           outputs='preprocessed_datasets')
    create_model_node = node(create_model, name='create_object_detection_model', inputs=['preprocessed_datasets', 'params:object_detector_model'], outputs='model')
    train_node = node(train, name='train_object_detector', inputs=['preprocessed_datasets', 'model', 'params:train_object_detector'], outputs='ignite_state')
    # hp_search_node = node(deepcv.meta.hyperparams.hp_search, name='hpsearch_object_detector',
    #                      inputs=['preprocessed_datasets', 'model', 'params:hpsearch_object_detector'], outputs='hpsearch_results')

    return {'preprocess_cifar': Pipeline([preprocess_node], tags=['preprocess']),
            'train_object_detector': Pipeline([preprocess_node, create_model_node, train_node], tags=['train', 'detection']),
            }  # 'hpsearch_object_detector':  Pipeline([preprocess_node, create_model_node, hp_search_node], tags=['train', 'hp_search', 'detection'])}


def create_model(datasets: Dict[str, Dataset], model_params: Union[deepcv.meta.hyperparams.Hyperparameters, Dict[str, Any]]):
    """ Creates object detection model from model specification of parameters.yml and determine input and output shapes according to dataset image shape and target classes count or shape """
    dummy_img, dummy_target = datasets['trainset'][0]
    input_shape = dummy_img.shape
    if not hasattr(model_params['architecture'][-1]['fully_connected'], 'out_features'):
        # Architecture's last FC output layer doesn't specify output features size, so we deduce it from dataset's target classes or target shape
        classes = deepcv.utils.recursive_getattr(datasets['trainset'], 'classes', recurse_on_type=Dataset)
        if classes is not None:
            model_params['architecture'][-1]['fully_connected']['out_features'] = len(classes)
        elif isinstance(dummy_target, torch.Tensor):
            model_params['architecture'][-1]['fully_connected']['out_features'] = np.prod(dummy_target.shape)

    # Create model according to its achritecture specification and return it
    model = deepcv.meta.base_module.DeepcvModule(input_shape, model_params)
    return model


def train(datasets: Dict[str, Dataset], model: nn.Module, hp: Union[deepcv.meta.hyperparams.Hyperparameters, Dict[str, Any]]) -> ignite.engine.State:
    """ Train object detector model using `deepcv.meta.ignite_training`
    .. See Fastai blog post about AdamW for more details about optimizer: https://www.fast.ai/2018/07/02/adam-weight-decay/
    """
    backend_conf = deepcv.meta.ignite_training.BackendConfig(**(hp['backend_conf'] if 'backend_conf' in hp else {}))
    metrics = {'accuracy': Accuracy(device=backend_conf.device if backend_conf.distributed else None)}
    loss = nn.CrossEntropyLoss()
    opt = optim.AdamW

    return deepcv.meta.ignite_training.train(hp, model, loss, datasets, opt, backend_conf, metrics)


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
