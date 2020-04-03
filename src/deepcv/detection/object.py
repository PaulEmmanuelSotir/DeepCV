#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Object detection module - object.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Any, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader

from ....tests.tests_utils import test_module
import deepcv.meta.ignite_training as training
from deepcv.meta.data.preprocess import preprocess_cifar

__all__ = []
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


class ObjectDetector(nn.Module):
    def __init__(self):
        super(ObjectDetector, self).__init__(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def get_object_detector_pipelines():
    preprocess_node = node(preprocess_cifar, name='preprocess_cifar_dataset', inputs=['cifar10_dataset', 'params:cifar10_testset_ratio'], outputs=['trainset', 'testset'])
    p1 = Pipeline([preprocess_node,
                   node(create_model, name='create_object_detection_model', inputs=['trainset', 'testset'], outputs=['trainset', 'testset', 'model']),
                   node(train, name='train_object_detector', inputs=['trainset', 'testset', 'params:object_detector_hp'], outputs=None)],
                  name='object_detector_training')
    return [p1]


def create_model(hp: Dict[str, Any]):
    model = ObjectDetector()
    return model


def load_dataset(, distributed: bool = False, hp: Dict[str, Any]):
    batch_size = hp['batch_size'] // (dist.get_world_size() if distributed else 1)


def train(trainset: DataLoader, validset: DataLoader, model: nn.Module, hp: Dict[str, Any]):
    metrics = {'accuracy': Accuracy(device=device if distributed else None)}
    device = utils.get_device(devid=hp['local_rank'])
    backend_conf = training.BackendConfig(device, hp.get('dist_backend'), hp['local_rank'])

    # TODO: remove these lines and create respective nodes
    num_workers = max(1, (ncpu_per_node - 1) // torch.cuda.device_count()) if torch.cuda.device_count() > 0 else max(1, multiprocessing.cpu_count() - 1)
    trainset, validset = get_train_test_loader(path=hp['data_path'], batch_size=batch_size, distributed=distributed, num_workers=num_workers)

    return training.train(hp, model, trainset, validset, distributed=distributed, device=device, metrics=metrics)


if __name__ == '__main__':
    test_module(__file__)
