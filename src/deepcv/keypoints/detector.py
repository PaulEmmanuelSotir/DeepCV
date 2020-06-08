#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Image/Video feature keypoints detection module - keypoints.detector.py - `DeepCV`__  
Implements image/video keypoint detector from [detector part of Unsupervised Learning of Object Structure and Dynamics from Videos](https://arxiv.org/pdf/1906.07889.pdf), [Google Research's official implentation](https://github.com/google-research/google-research/tree/master/video_structure) is based on Tensorflow, thus, we had to implement it ourselves.  
.. moduleauthor:: Paul-Emmanuel Sotir  

*TODO List*  
    - TODO: Video interpolation / dynamics learning and also relevant for unsupervised keypoint detection ideas: https://github.com/google-research/google-research/tree/master/video_structure from this paper: https://arxiv.org/abs/1906.07889   
    - TODO: NN model for Keypoints proposal from a conv NN which outputs K feature maps: each output channel is normalized and averaged into (x,y) coordinates in order to obtain relevant keypoints (K at most). Trained using a autoencoder setup: a generator (decoder with end-to-end skip connection from anchor frame) must be able to reconstruct input image from keypoints (converted to gaussian heat maps) and another frame along with its own keypoints (e.g. first video frame)  
    - TODO: Modify keypoint model in order to have feature pattern information associated with keypoint coordinates (instead of simply associate input image)?  
"""
import logging
import multiprocessing
from typing import Union, List, Dict, Any, Optional

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import ignite
import numpy as np
from ignite.metrics import Accuracy
from kedro.pipeline import Pipeline, node

import deepcv.utils
from deepcv import meta
from deepcv.meta import base_module
from deepcv.meta import hyperparams

__all__ = ['KeypointsDetector']
__author__ = 'Paul-Emmanuel Sotir'


class KeypointsDetector(base_module.DeepcvModule):
    HP_DEFAULTS = {'architecture': ..., 'act_fn': nn.ReLU, 'batch_norm': None, 'dropout_prob': 0.}

    def __init__(self, input_shape: torch.Tensor, hp: Union[Dict[str, Any], hyperparams.Hyperparameters]):
        super().__init__(input_shape, hp)
        self.define_nn_architecture(self._hp['architecture'])
        self.initialize_parameters(self._hp['act_fn'])


def get_keypoints_detector_pipelines():
    p1 = Pipeline([node(deepcv.meta.data.preprocess.split_dataset, name='split_dataset',
                        inputs={'trainset': 'cifar10_train', 'testset': 'cifar10_test', 'params': 'params:split_dataset_ratios'},
                        outputs=['trainset', 'validset', 'testset']),
                   node(deepcv.meta.data.preprocess.preprocess, name='preprocess',
                        inputs={'trainset': 'trainset', 'testset': 'testset', 'validset': 'validset', 'params': 'params:cifar10_preprocessing'},
                        outputs=['datasets']),
                   node(create_model, name='create_keypoints_encoder_model', inputs=['datasets', 'params:keypoints_encoder_model'], outputs=['encoder']),
                   node(create_model, name='create_keypoints_decoder_model', inputs=['datasets', 'params:keypoints_decoder_model'], outputs=['decoder']),
                   node(train, name='train_object_detector', inputs=['datasets', 'model', 'params:train_object_detector'], outputs=['ignite_state'])],
                  name='train_object_detector')
    return [p1]


def create_model(datasets: Dict[str, Dataset], model_params: Union[hyperparams.Hyperparameters, Dict[str, Any]]):
    # Determine input and output shapes
    dummy_img, dummy_target = datasets['train_loader'][0][0]
    input_shape = dummy_img.shape
    model_params['architecture'][-1]['fully_connected']['out_features'] = np.prod(dummy_target.shape)

    # Create ObjectDetector model
    model = KeypointsDetector(input_shape, model_params)
    return model


def train(datasets: Dict[str, Dataset], encoder: nn.Module, decoder: nn.Module, hp: Union[hyperparams.Hyperparameters, Dict[str, Any]]) -> ignite.engine.State:
    # TODO: decide whether we take Datasets or Dataloaders arguments here (depends on how preprocessing is implemented)
    backend_conf = deepcv.meta.ignite_training.BackendConfig(**hp['backend_conf'])
    metrics = {'accuracy': Accuracy(device=backend_conf.device if backend_conf.distributed else None)}
    loss = nn.CrossEntropyLoss()
    opt = optim.SGD

    # Determine maximal eval batch_size which fits in video memory
    max_eval_batch_size = deepcv.meta.nn.find_best_eval_batch_size(datasets['trainset'][0].shape, model=model, device=backend_conf.device, upper_bound=len(datasets['trainset']))

    # Determine num_workers for DataLoaders
    if backend_conf.ngpus_current_node > 0 and backend_conf.distributed:
        workers = max(1, (backend_conf.ncpu - 1) // backend_conf.ngpus_current_node)
    else:
        workers = max(1, multiprocessing.cpu_count() - 1)

    # Create dataloaders from dataset
    dataloaders = []
    for n, ds in datasets.items():
        shuffle = True if n == 'trainset' else False
        batch_size = hp['batch_size'] if n == 'trainset' else max_eval_batch_size
        dataloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True))

    return deepcv.meta.ignite_training.train(hp, model, loss, dataloaders, opt, backend_conf, metrics)


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
