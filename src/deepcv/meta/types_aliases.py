#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Typing meta module - submodules_creators.py - `DeepCV`__
Type aliases and generics for easier and consistent type checking and type annotations across DeepCV
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import enum
import types
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence, List, Set, Mapping

import torch
import torch.nn
from torch.utils.data import Dataset

import deepcv.utils

__all__ = ['TENSOR_OR_SEQ_OF_TENSORS_T', 'FLOAT_OR_FLOAT_TENSOR_T', 'HYPERPARAMS_T', 'METRICS_DICT_T', 'SUBMODULE_FORWARD_CALLBACK_T',
           'SUBMODULE_CREATORS_DICT_T', 'REDUCTION_FN_T', 'NORM_TECHNIQUES_MODULES_T', 'MODULE_OR_TYPE_T',
           'SIZE_1_T', 'SIZE_2_T', 'SIZE_3_T', 'SIZE_N_T']
__author__ = 'Paul-Emmanuel Sotir'

#______________________________________________ TYPES ALIASES CONSTANTS _______________________________________________#


TENSOR_OR_SEQ_OF_TENSORS_T = Union[torch.Tensor, Sequence[torch.Tensor]]

FLOAT_OR_FLOAT_TENSOR_T = Union[float, torch.FloatTensor]

HYPERPARAMS_T = Union['deepcv.meta.hyperparams.Hyperparameters', Dict[str, Any]]

# Here, `METRICS_DICT_T` is the result(s) from evaluation of named metric(s) (Not a map of named metric function(s))
METRICS_DICT_T = Mapping[str, FLOAT_OR_FLOAT_TENSOR_T]

LOSS_FN_T = Union[torch.nn.modules.loss._Loss, Callable[['target_pred', 'target', ...], Union[float, torch.Tensor]]]

# Unlike losses, we may assume that a metric (`deepcv.meta.types_aliases.METRIC_FN_T`) allays returns a single value (allays reduces over minibatch dim)
METRIC_FN_T = Union[ignite.Metric, LOSS_FN_T]

# `TRAINING_PROCEDURE_T` training procedures must match the following type annotation for its input arguments: `'hp': HYPERPARAMS_T, 'model': torch.nn.Module, 'loss': LOSS_FN_T, 'datasets': Tuple[Dataset], 'opt': Type[torch.optim.Optimizer], 'backend_conf': 'deepcv.meta.ignite_training.BackendConfig', 'metrics': Dict[str, METRIC_FN_T], 'callbacks_handler': Optional[deepcv.utils.CallbacksHandler], **kwargs`
TRAINING_PROCEDURE_T = Callable[['hp', 'model', 'loss', 'datasets', 'opt', 'backend_conf', 'metrics', 'callbacks_handler', ...], Tuple[METRICS_DICT_T, ...]]

SUBMODULE_FORWARD_CALLBACK_T = Callable[[TENSOR_OR_SEQ_OF_TENSORS_T, Dict[str, torch.Tensor]], TENSOR_OR_SEQ_OF_TENSORS_T]

SUBMODULE_CREATORS_DICT_T = Dict[str, Callable[..., torch.nn.Module]]

REDUCTION_FN_T = Callable[[TENSOR_OR_SEQ_OF_TENSORS_T, *'args'], TENSOR_OR_SEQ_OF_TENSORS_T]

NORM_TECHNIQUES_MODULES_T = Dict['NormTechnique', Union[Type[torch.nn.Module], Callable[..., torch.nn.Module]]]

MODULE_OR_TYPE_T = Union[torch.nn.Module, Type[torch.nn.Module]]

SIZE_1_T = Union[int, Tuple[int]]  # e.g. can be used for 1D conv kernel size, padding, ... (equivalent to torch.nn.common_types._size_1_t)
SIZE_2_T = Union[int, Tuple[int, int]]  # e.g. can be used for 2D conv kernel size, padding, ... (equivalent to torch.nn.common_types._size_2_t)
SIZE_3_T = Union[int, Tuple[int, int, int]]  # e.g. can be used for 3D conv kernel size, padding, ... (equivalent to torch.nn.common_types._size_3_t)
SIZE_N_T = Union[int, Tuple[int, ...]]  # Size type on N-D operations (equivalent to torch.nn.common_types._size_any_t)

#______________________________________________ TYPES ALIASES UNIT TESTS ______________________________________________#


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
