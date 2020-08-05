#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Image classification module - classification.image.py - `DeepCV`__  
Simple image classification model training, hp search and Neural-Architecture-Search on CIFAR-10/CIFAR-A00 dataset.  
.. moduleauthor:: Paul-Emmanuel Sotir  
"""
from typing import Any, Dict, Optional, Tuple, Callable, List, Iterable, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import ignite
import numpy as np
from ignite.metrics import Accuracy
from kedro.pipeline import Pipeline, node

import deepcv.utils
import deepcv.meta.nn
import deepcv.meta.base_module
import deepcv.meta.nni_tools
import deepcv.meta.hyperparams
import deepcv.meta.data.preprocess
import deepcv.meta.ignite_training
from deepcv.meta.data.datasets import batch_prefetch_dataloader_patch
from deepcv.meta.types_aliases import *

__all__ = ['get_img_classifier_pipelines', 'create_model', 'train']
__author__ = 'Paul-Emmanuel Sotir'


def get_img_classifier_pipelines() -> Dict[str, Pipeline]:
    """ Defines all Kedro nodes and pipelines of image classifier model: datasets setup/preprocessing, image classifier training and image classifier hyperparameters search pipelines. """
    preprocess_node = node(deepcv.meta.data.preprocess.preprocess, name='preprocess',
                           inputs=dict(dataset_or_trainset='cifar10_train', testset='cifar10_test', params='params:cifar10_preprocessing'),
                           outputs='preprocessed_datasets')
    create_model_node = node(create_model, name='create_image_classifier_model', inputs=['preprocessed_datasets', 'params:image_classifier_model'], outputs='model', tags=['model'])
    train_node = node(train, name='train_image_classifier', inputs=['preprocessed_datasets', 'model', 'params:train_image_classifier'], outputs='ignite_state')

    # Hyperparameters search pipeline nodes
    nni_hp_search_sample_node = node(deepcv.meta.nni_tools.sample_nni_hp_space, name='sample_nni_hp_search_space',
                                     inputs=dict(yml_parameters='parameters'),
                                     outputs=['model_hps', 'training_hps'])
    create_model_node = node(create_model, name='create_image_classifier_model', inputs=['preprocessed_datasets', 'model_hps'], outputs='model')
    train_node = node(train, name='train_image_classifier', inputs=['preprocessed_datasets', 'model', 'training_hps'], outputs='ignite_state')

    return {'preprocess_cifar': Pipeline([preprocess_node], tags=['preprocess']),
            'train_image_classifier': Pipeline([preprocess_node, create_model_node, train_node], tags=['train', 'classification']),
            'hp_search_image_classifier': Pipeline([nni_hp_search_sample_node, preprocess_node, create_model_node, train_node], tags=['train', 'classification', 'hp_search'])}  # 'hpsearch_image_classifier':  Pipeline([preprocess_node, create_model_node, hp_search_node], tags=['train', 'hp_search', 'classification'])}


def create_model(datasets: Dict[str, Dataset], model_params: HYPERPARAMS_T):
    """ Creates image classifier model from model specification of parameters.yml and determine input and output shapes according to dataset image shape and target classes count or shape """
    dummy_img, dummy_target = datasets['trainset'][0]
    input_shape = dummy_img.shape
    if not hasattr(model_params['architecture'][-1]['fully_connected'], 'out_features'):
        # Architecture's last FC output layer doesn't specify output features size, so we deduce it from dataset's target classes or target shape
        classes = deepcv.utils.recursive_getattr(datasets['trainset'], 'classes', recurse_on_type=Dataset)
        if classes is not None:
            model_params['architecture'][-1]['fully_connected']['out_features'] = len(classes)
        elif isinstance(dummy_target, torch.Tensor):
            model_params['architecture'][-1]['fully_connected']['out_features'] = np.prod(dummy_target.shape)

    # Create model according to its architecture specification and return it
    model = deepcv.meta.base_module.DeepcvModule(input_shape, model_params)
    return model



def train(datasets: Dict[str, Dataset], model: nn.Module, hp: HYPERPARAMS_T) -> Tuple[METRICS_DICT_T, Union['final_nas_architecture_path', ignite.engine.State], Optional[str]]:
    """ Train image classifier model using `deepcv.meta.ignite_training`
    .. See Fastai blog post about AdamW for more details about optimizer: https://www.fast.ai/2018/07/02/adam-weight-decay/
    """
    backend_conf = deepcv.meta.ignite_training.BackendConfig(**(hp['backend_conf'] if 'backend_conf' in hp else {}))
    training_procedure_kwargs = dict(hp=hp, model=model, datasets=datasets, backend_conf=backend_conf,
                                     loss=nn.CrossEntropyLoss(),
                                     opt=optim.AdamW,
                                     metrics={'accuracy': Accuracy(device=backend_conf.device if backend_conf.distributed else None)},
                                     callbacks_handler=None)
    single_shot_nas = ...  # TODO: ...
    if single_shot_nas:
        fixed_architecture_path = ...
        return deepcv.meta.nni_tools.nni_single_shot_neural_architecture_search(**training_procedure_kwargs, final_architecture_path=fixed_architecture_path)
    else:
        nni_compression_pruner = None
        return (*deepcv.meta.ignite_training.train(**training_procedure_kwargs, nni_compression_pruner=nni_compression_pruner), None)


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
