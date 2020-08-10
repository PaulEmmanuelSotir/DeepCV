#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Image classification module - classification.image.py - `DeepCV`__  
Simple image classification model training, hp search and Neural-Architecture-Search on CIFAR-10/CIFAR-A00 dataset.  
.. moduleauthor:: Paul-Emmanuel Sotir  
"""
from typing import Any, Dict, Optional, Tuple, Callable, List, Iterable, Union

import torch
import torch.nn
import torch.optim
from torch.utils.data import Dataset

import ignite
import numpy as np
from ignite.metrics import Accuracy
from kedro.pipeline import Pipeline, node

import deepcv.utils
import deepcv.meta
import deepcv.meta.data.preprocess
from deepcv.meta.types_aliases import HYPERPARAMS_T, METRICS_DICT_T

__all__ = ['get_img_classifier_pipelines', 'create_model', 'train']
__author__ = 'Paul-Emmanuel Sotir'


def get_img_classifier_pipelines() -> Dict[str, Pipeline]:
    """ Defines all Kedro nodes and pipelines of image classifier model: datasets setup/preprocessing, image classifier training and image classifier hyperparameters search pipelines. """
    preprocess_node = node(deepcv.meta.data.preprocess.preprocess, name='preprocess', tags=['preprocess'],
                           inputs=dict(dataset_or_trainset='cifar10_train', testset='cifar10_test', params='params:cifar10_preprocessing'),
                           outputs='datasets')
    create_model_node = node(create_model, name='create_image_classifier_model', inputs=['datasets', 'params:image_classifier_model'], outputs='model', tags=['model'])
    train_node = node(train, name='train_image_classifier', inputs=['datasets', 'model', 'params:train_image_classifier'], outputs='train_results', tags=['train'])

    return {'preprocess_cifar': Pipeline([preprocess_node]),
            'train_image_classifier': Pipeline([preprocess_node, create_model_node, train_node], tags=['classification'])}


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
    return deepcv.meta.base_module.DeepcvModule(input_shape, model_params)


# @image_classifier.single_shot_nas_training_procedure(allow_hp_search=True)
# def train_single_shot_nas(datasets: Dict[str, Dataset], model: torch.nn.Module, hp: HYPERPARAMS_T) -> METRICS_DICT_T:
#     pass
#     # TODO: ... deepcv.meta.nni_tools.handle_nni_nas_trial


# @image_classifier.training_procedure(allow_hp_search=True, allow_classic_nas=True)

def train(datasets: Dict[str, Dataset], model: torch.nn.Module, hp: HYPERPARAMS_T) -> Tuple[METRICS_DICT_T, Union['final_nas_architecture_path', ignite.engine.State], Optional[str]]:
    """ Train image classifier model using `deepcv.meta.ignite_training`
    .. See Fastai blog post about AdamW for more details about optimizer: https://www.fast.ai/2018/07/02/adam-weight-decay/
    """
    backend_conf = deepcv.meta.ignite_training.BackendConfig(**(hp['backend_conf'] if 'backend_conf' in hp else {}))
    training_procedure_kwargs = dict(hp=hp, model=model, datasets=datasets, backend_conf=backend_conf,
                                     loss=torch.nn.CrossEntropyLoss(),
                                     opt=torch.optim.AdamW,
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
