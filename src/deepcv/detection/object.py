#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Object detection module - object.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Any, Dict, Optional, Tuple, Callable
from pathlib import Path
import multiprocessing

from kedro.pipeline import Pipeline, node

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy

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

# TODO: refactor to handle test and/or valid sets + handle cross validation
# TODO: add support for residual/dense links
# TODO: make this model fully generic and move it to meta.nn (allow module_creators to be extended and/or overriden)


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


def _create_fully_connected(layer_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: Dict[str, Any]):
    layer_params['in_features'] = np.prod(prev_shapes[-1][1:])  # We assume here that features/inputs are given in batches
    if 'out_features' not in layer_params:
        # Handle last fully connected layer (no dropout nor batch normalization for this layer)
        layer_params['out_features'] = self._output_size
        layers.append(tu.fc_layer(layer_params))
    else:
        layers.append(tu.fc_layer(layer_params, hp['act_fn'], hp['dropout_prob'], hp['batch_norm']))
    prev_out = layer_params['out_features']


def _create_flatten(*_args, **_kwargs):
    return tu.Flatten()


def _initialize_weights(module: nn.Module):
    if meta.nn.is_conv(module):
        nn.init.xavier_normal_(module.weight.data, gain=self._xavier_gain)
        module.bias.data.fill_(0.)
    elif utils.is_fully_connected(module):
        nn.init.xavier_uniform_(module.weight.data, gain=self._xavier_gain)
        module.bias.data.fill_(0.)
    elif type(module).__module__ == nn.BatchNorm2d.__module__:
        nn.init.uniform_(module.weight.data)  # gamma == weight here
        module.bias.data.fill_(0.)  # beta == bias here
    elif list(module.parameters(recurse=False)) and list(module.children()):
        raise Exception("ERROR: Some module(s) which have parameter(s) haven't bee explicitly initialized.")


class ObjectDetector(meta.nn.DeepcvModule):
    HP_DEFAULTS = {'architecture': ..., 'act_fn': nn.ReLU, 'batch_norm': None, 'dropout_prob': 0.}

    def __init__(self, input_shape: torch.Size, output_size: torch.Size, module_creators: Dict[str, Callable], hp: Dict[str, Any], init_params_fn=None):
        super(ObjectDetector, self).__init__(input_shape, hp)
        self._output_size = output_size
        self._xavier_gain = nn.init.calculate_gain(tu.get_gain_name(hp['act_fn']))
        self._features_shapes = []
        self._architecture = list(hp['architecture'])

        # Define neural network architecture
        modules = []
        for name, params in self._architecture:
            # Create layer/block module
            fn = module_creators.get(name)
            if not fn:
                raise RuntimeError(f'Error: Could not create layer/module named "{name}" given module creators: "{_LAYER_CREATORS.keys()}"')
            module = fn(params, self._features_shapes, hp)
            modules.append(module)

            # Get neural network output features shapes by performing a dummy forward
            with torch.no_grad():
                dummy_batch_x = torch.unsqueeze(torch.zeros(self._input_shape), dim=0)
                self._features_shapes.append(nn.Sequential(*modules)(dummy_batch_x).shape)
        self._net = nn.Sequential(*modules)

        if init_params_fn:
            self.apply(init_params_fn)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._net(x)  # Apply whole neural net architecture

    def __str__(self) -> str:
        capacity = utils.human_readable_size(meta.nn.get_model_capacity(self))
        modules_str = '\n\t'.join([f'- {n}({p}) output_features_shape={s}' for (n, p), s in zip(self._architecture, self._features_shapes)])
        return f'{self.__class__.__name__} (capacity={capacity}):\n\t{modules_str}'


def get_object_detector_pipelines():
    p1 = Pipeline([node(utils.merge_dicts, name='merge_hyperparameters', inputs=['params:ignite_training', 'params:object_detector', 'params:cifar10'], outputs=['hp']),
                   node(meta.data.preprocess.preprocess_cifar, name='preprocess_cifar_dataset', inputs=['cifar10_train', 'cifar10_test', 'hp'], outputs=['dataset']),
                   node(create_model, name='create_object_detection_model', inputs=['dataset', 'hp'], outputs=['model']),
                   node(train, name='train_object_detector', inputs=['dataset', 'model', 'hp'], outputs=None)],
                  name='object_detector_training')
    return [p1]


def create_model(trainset: DataLoader, hp: Dict[str, Any]):
    dummy_img, dummy_target = trainset[0][0]
    input_shape = dummy_img.shape
    output_size = dummy_target.shape
    module_creators = {'avg_pooling': _create_avg_pooling, 'conv2d': _create_conv2d, 'fully_connected': _create_fully_connected, 'flatten': _create_flatten}
    model = ObjectDetector(input_shape, output_size, module_creators, hp['model'], init_params_fn=_initialize_weights)
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
    dataloaders = (DataLoader(ds, hp['batch_size'], shuffle=True if i == 0 else False, num_workers=workers) for i, ds in enumerate(datasets))

    return meta.ignite_training.train(hp, model, loss, dataloaders, opt, backend_conf, metrics)


if __name__ == '__main__':
    test_module(__file__)
