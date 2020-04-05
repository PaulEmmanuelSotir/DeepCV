#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Object detection module - object.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Any, Dict, Optional
from pathlib import Path

from kedro.pipeline import Pipeline
from kedro.node import node

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader

from ...tests.tests_utils import test_module
import deepcv.meta as meta
import deepcv.utils as utils

__all__ = ['ObjectDetector', 'get_object_detector_pipelines', 'load_dataset', 'create_model', 'train']
__author__ = 'Paul-Emmanuel Sotir'

"""
'seed': 563454 (used only if deterministic is True)
'deterministic': False
'local_rank': 0
'dist_url': # "env://"
'dist_backend': None # Set, for example, 'nccl' (torch.distributed.Backend.NCCL) for distributed training using nccl backend
'batch_size':
'optimizer': optim.SGD
'optimizer_opts': {'lr', 'momentum':, 'weight_decay', 'nesterov': True}
'loss': nn.CrossEntropyLoss()
'shceduler_milestones_values': [(0, 0.0),
                                (len(trainset) * hp['warmup_epochs'], hp['optimizer_opts']['lr']),
                                (len(trainset) * hp['epochs'], 0.0)]}
"""
# TODO: implement a new node ('load_dataset') instead of get_train_test_loader
# TODO: implement a 'define_model' node instead of get_model
# TODO: add schduler to hp
# TODO: refactor to handle test and/or valid sets + handle cross validation
# TODO: refactor to handle given metrics


class ObjectDetector(meta.nn.DeepcvModule):
    HP_DEFAULTS = {'layers': ..., 'batch_norm': None, 'dropout_prob': 0.}

    def __init__(self, input_shape, hp):
        super(ObjectDetector, self).__init__(input_shape, hp)
        self.net = nn.Sequential()
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def get_object_detector_pipelines():
    p1 = Pipeline([node(utils.merge_dicts, name='merge_hyperparameters', inputs=['params:ignite_training', 'params:object_detector', 'params:cifar10'], outputs=['hp']),
                   node(meta.data.preprocess.preprocess_cifar, name='preprocess_cifar_dataset', inputs=['cifar10_train', 'cifar10_test', 'hp'], outputs=['dataset']),
                   node(create_model, name='create_object_detection_model', inputs=['dataset', 'hp'], outputs=['model']),
                   node(train, name='train_object_detector', inputs=['dataset', 'model', 'hp'], outputs=None)],
                  name='object_detector_training')
    return [p1]


def load_dataset(hp: Dict[str, Any]):
    if distributed:
        hp['batch_size'] = hp['batch_size'] // dist.get_world_size()


def create_model(trainset: DataLoader, hp: Dict[str, Any]):
    dummy_img = trainset[0][0]
    input_shape = dummy_img.shape
    model = ObjectDetector(input_shape, hp)
    return model


def train(dataset: Tuple[DataLoader], model: nn.Module, hp: Dict[str, Any]):
    device, backend_conf = hp['device'], hp['backend_conf']
    metrics = {'accuracy': Accuracy(device=device if distributed else None)}
    loss = nn.CrossEntropyLoss()
    opt = optim.SGD

    # TODO: remove these lines and create respective nodes
    trainset, validset = get_train_test_loader(path=hp['data_path'], batch_size=batch_size, distributed=backend_conf.distributed, num_workers=num_workers)

    return meta.ignite_training.train(hp, model, loss, dataset, opt, backend_conf, device, metrics)


if __name__ == '__main__':
    test_module(__file__)
