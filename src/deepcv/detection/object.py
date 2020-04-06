#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Object detection module - object.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Any, Dict, Optional, Tuple
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


def _create_avg_pooling(layer_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: Dict[str, Any]):
    prev_dim = len(prev_shapes[-1])
    if prev_dim >= 4:
        return nn.AvgPool3d(**layer_params)
    elif prev_dim >= 2:
        return nn.AvgPool2d(**layer_params)
    return nn.AvgPool1d(**layer_params)


def _create_conv2d(layer_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: Dict[str, Any]):
    layer_params['in_channels'] = prev_shapes[-1][0]
    layers.append(tu.conv_layer(layer_params, hp['act_fn'], hp['dropout_prob'], hp['batch_norm']))
    prev_out = layer_params['out_channels']
    # Get convolution output features shape by performing a dummy forward
    with torch.no_grad():
        net = nn.Sequential(*layers)
        dummy_batch_x = torch.zeros(torch.Size((1, *self._input_shape)))
        self._conv_features_shapes.append(net(dummy_batch_x).shape)


def _create_fully_connected(layer_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: Dict[str, Any]):
    # Determine in_features for this fully connected layer
    if in_conv_backbone:  # First FC layer following convolution backbone
        self._conv_out_features = np.prod(self._conv_features_shapes[-1][-3:])
        layer_params['in_features'] = self._conv_out_features
        in_conv_backbone = False
    else:
        layer_params['in_features'] = prev_out
    if 'out_features' not in layer_params:
        # Handle last fully connected layer (no dropout nor batch normalization for this layer)
        layer_params['out_features'] = self._output_size
        layers.append(tu.fc_layer(layer_params))
    else:
        layers.append(tu.fc_layer(layer_params, hp['act_fn'], hp['dropout_prob'], hp['batch_norm']))
    prev_out = layer_params['out_features']


def _create_flatten(*_args, **_kwargs):
    return tu.Flatten()


_LAYER_CREATORS = {'avg_pooling': _create_avg_pooling, 'conv2d': _create_conv2d, 'fully_connected': _create_fully_connected, 'flatten': _create_flatten}


class ObjectDetector(meta.nn.DeepcvModule):
    HP_DEFAULTS = {'architecture': ..., 'act_fn': nn.ReLU, 'batch_norm': None, 'dropout_prob': 0.}

    def __init__(self, input_shape: torch.Size, output_size: torch.Size, hp: Dict[str, Any]):
        super(ObjectDetector, self).__init__(input_shape, hp)
        self._output_size = output_size
        self._xavier_gain = nn.init.calculate_gain(tu.get_gain_name(hp['act_fn']))
        self._conv_features_shapes, self._conv_out_features = [], None  # TODO: refactor this to be 'prev_shapes'
        hp['architecture'] = list(hp['architecture'])

        # Define neural network architecture
        layers = []
        in_conv_backbone, prev_out = True, self._input_shape  # TODO: replace these vars with 'prev_shapes'

        for name, params in hp['architecture']:
            fn = _LAYER_CREATORS.get(name)
            if not fn:
                raise RuntimeError(f'Error: Could not create layer/module named "{name}" given module creators: "{_LAYER_CREATORS.keys()}"')
            module, prev_shapes = fn(params, prev_shapes, hp)
            self._layers.append(module)
        self._layers = nn.Sequential(*layers)

    def init_params(self):
        def _initialize_weights(module: nn.Module):
            if meta.nn.is_conv(module):
                nn.init.xavier_normal_(module.weight.data, gain=self._xavier_gain)
                module.bias.data.fill_(0.)
            elif utils.is_fully_connected(module):
                nn.init.xavier_unifor                                                                                               m_(module.weight.data, gain=self._xavier_gain)
                module.bias.data.fill_(0.)
            elif type(module).__module__ == nn.BatchNorm2d.__module__:
                nn.init.uniform_(module.weight.data)  # gamma == weight here
                module.bias.data.fill_(0.)  # beta == bias here
            elif list(module.parameters(recurse=False)) and list(module.children()):
                raise Exception("ERROR: Some module(s) which have parameter(s) haven't bee explicitly initialized.")
        self.apply(_initialize_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._layers(x)  # Apply whole neural net architecture


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
    output_size = ???  # TODO: !!
    model = ObjectDetector(input_shape, output_size, **hp['model'])
    model.init_params()
    return model


def train(datasets: Tuple[torch.utils.data.Dataset], model: nn.Module, hp: Dict[str, Any]):
    backend_conf = meta.ignite_training.BackendConfig(dist_backend=hp['dist_backend'], dist_url=hp['dist_url'], local_rank=hp['local_rank'])
    metrics = {'accuracy': Accuracy(device='cuda:{backend_conf.local_rank}' if backend_conf.distributed else None)}
    loss = nn.CrossEntropyLoss()
    opt = optim.SGD

    # Create dataloaders from dataset
    if backend_conf.ngpus_current_node > 0 and backend_conf.distributed:
        workers = max(1, (backend_conf.ncpu_per_node - 1) // backend_conf.ngpus_current_node)
    else:
        workers = max(1, multiprocessing.cpu_count() - 1)
    dataloaders = (DataLoader(ds, TODO..., workers) for ds in datasets)

    # TODO: remove these lines and create respective nodes
    trainset, validset = get_train_test_loader(path=hp['data_path'], batch_size=batch_size, distributed=backend_conf.distributed, num_workers=num_workers)

    return meta.ignite_training.train(hp, model, loss, dataloaders, opt, backend_conf, metrics)


if __name__ == '__main__':
    test_module(__file__)
