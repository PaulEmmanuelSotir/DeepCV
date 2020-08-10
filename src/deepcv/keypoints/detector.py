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
import torch.nn
import torch.optim
from torch.utils.data import Dataset

import ignite
import numpy as np
from ignite.metrics import Accuracy
from kedro.pipeline import Pipeline, node

import deepcv.utils
import deepcv.meta
from deepcv.meta.types_aliases import HYPERPARAMS_T, METRICS_DICT_T
import deepcv.meta.data.preprocess

__all__ = ['KeypointsDetector']
__author__ = 'Paul-Emmanuel Sotir'


class KeypointsDetector(deepcv.meta.base_module.DeepcvModule):
    HP_DEFAULTS = {'architecture': ..., 'act_fn': nn.ReLU, 'batch_norm': None, 'dropout_prob': 0.}

    def __init__(self, input_shape: torch.Tensor, hp: HYPERPARAMS_T):
        super().__init__(input_shape, hp)
        self.define_nn_architecture(self._hp['architecture'])
        self.initialize_parameters(self._hp['act_fn'])


def get_keypoints_detector_pipelines():
    p1 = Pipeline([node(deepcv.meta.data.preprocess.preprocess, name='preprocess', tags=['preprocess'],
                        inputs={'dataset_or_trainset': 'cifar10_train', 'testset': 'cifar10_test', 'params': 'params:cifar10_preprocessing'},
                        outputs=['datasets']),
                   node(create_model, name='create_keypoints_encoder_model', inputs=['datasets', 'params:keypoints_encoder_model'], outputs=['encoder'], tags=['model']),
                   node(create_model, name='create_keypoints_decoder_model', inputs=['datasets', 'params:keypoints_decoder_model'], outputs=['decoder'], tags=['model']),
                   node(train, name='train_keypoint_detector', inputs=['datasets', 'encoder', 'decoder', 'params:train_keypoint_detector'], outputs=['train_results'], tags=['train'])],
                  tags=['keypoint', 'detection'])
    return {'train_keypoint_detector': p1}


def create_model(datasets: Dict[str, Dataset], model_params: HYPERPARAMS_T):
    # Determine input and output shapes
    dummy_img, dummy_target = datasets['train_loader'][0][0]
    input_shape = dummy_img.shape

    if not hasattr(model_params['architecture'][-1]['fully_connected'], 'out_features'):
        # TODO: modify output usage for inference on detection task
        model_params['architecture'][-1]['fully_connected']['out_features'] = np.prod(dummy_target.shape)

    # Create ImageClassifier model
    return deepcv.meta.base_module.DeepcvModule(input_shape, model_params)


def train(datasets: Dict[str, Dataset], encoder: nn.Module, decoder: nn.Module, hp: HYPERPARAMS_T) -> Tuple[METRICS_DICT_T, Union['final_nas_architecture_path', ignite.engine.State], Optional[str]]:
    # TODO: decide whether we take Datasets or Dataloaders arguments here (depends on how preprocessing is implemented)
    backend_conf = deepcv.meta.ignite_training.BackendConfig(**(hp['backend_conf'] if 'backend_conf' in hp else {}))
    autoencoder = torch.nn.Sequential(encoder, decoder)
    training_procedure_kwargs = dict(hp=hp, model=autoencoder, datasets=datasets, backend_conf=backend_conf,
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
