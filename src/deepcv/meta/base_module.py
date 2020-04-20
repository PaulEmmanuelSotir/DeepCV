#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" DeepCV model base class meta module - base_module.py - `DeepCV`__
Defines DeepCV model base class
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import types
import inspect
import logging
from collections import OrderedDict
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence

import torch
import torch.nn as nn

import numpy as np

import deepcv.meta as meta
import deepcv.utils as utils
from tests.tests_utils import test_module

__all__ = ['BASIC_SUBMODULE_CREATORS', 'DeepcvModule', 'DeepcvModuleDescriptor']
__author__ = 'Paul-Emmanuel Sotir'

BASIC_SUBMODULE_CREATORS = {'avg_pooling': _create_avg_pooling, 'conv2d': _create_conv2d, 'fully_connected': _create_fully_connected}


class DeepcvModule(nn.Module):
    """ DeepCV PyTorch Module model base class
    Handles hyperparameter defaults, NN architecture definition tooling and basic shared convolution block for transfert learning between all DeepCV models
    Child class must define `HP_DEFAULTS` class attribute, with at least the following keys: `{'architecture': ..., 'act_fn': ...}` and other needed hyperparameters deepending on which sub-module are specified in `architecture` definition
    For more details about `architecture` hyperparameter parsing, see code in `DeepcvModule._define_nn_architecture`.
    NOTE: in order `_features_shapes`, `_submodules_capacities` and `self._architecture_spec` attributes to be defined and contain NN sumbmodules informations, you need to call `DeepcvModule._define_nn_architecture` or update it by yourslef according to your NN architecture.
    NOTE: `self.__str__` outputs a human readable string describing NN's architecture with their respective feature_shape and capacity. In order to be accurate, you need to call `self._define_nn_architecture` or, alternatively, keep `_features_shapes` and `_submodules_capacities` attribute up-to-date and make sure that `self._architecture_spec` or self._hp['architecture'] contains architecture definition (similar value than `self._define_nn_architecture`'s `architecture_spec` argument would have).
    # TODO: implement basic conv block shared by all DeepcvModules (frozen weights by default, and allow forking of these weights to be specific to a given model) + update architecture_spec/features_shapes
    # TODO: move code from ObjectDetector into DeepcvModule
    """

    HP_DEFAULTS = ...
    SHARED_BLOCK_DISABLED_WARNING_MSG = r'Warning: `DeepcvModule.{}` called while `self._enable_shared_image_embedding_block` is `False` (Shared image embedding block disabled for this model)'

    def __init__(self, input_shape: torch.Size, hp: meta.hyperparams.Hyperparameters, enable_shared_block: bool = True, freeze_shared_block: bool = True):
        super(self.__class__).__init__(self)

        # Process module hyperparameters
        assert self.__class__.HP_DEFAULTS != ..., f'Error: Module classes which inherits from "DeepcvModule" ({self.__class__.__name__}) must define "HP_DEFAULTS" class attribute dict.'
        self._hp, missing_hyperparams = hp.with_defaults(self.__class__.HP_DEFAULTS)
        assert len(missing_hyperparams) > 0, f'Error: Missing required hyper-parameter in "{self.__class__.__name__}" module parameters. (missing: "{missing_hyperparams}")'

        self._input_shape = input_shape
        self._shared_block_forked = False
        self._enable_shared_image_embedding_block = enable_shared_block

        self.freeze_shared_image_embedding_block = freeze_shared_block
        if enable_shared_block and not 'shared_image_embedding_block' in self.__class__.__dict__:
            # If class haven't been instanciated yet, define common/shared DeepcvModule image embedding block
            self.__class__._define_shared_image_embedding_block()

    def __str__(self) -> str:
        """ Describes DeepCV module in a human readable text string, see `DeepcvModule.describe()` function or `DeepcvModuleDescriptor` class for more details """
        return str(self.describe())

    def describe(self) -> DeepcvModuleDescriptor:
        """ Describes DeepCV module with its architecture, capacity and features shapes at sub-modules level.
        Args:
            - to_string: Whether deepcv NN module should be described by a human-readable text string or a NamedTuple of various informations which, for example, makes easier to visualize model's sub-modules capacities or features shapes...
        Returns a `DeepcvModuleDescriptor` which contains model name, capacity, and eventually submodules names, feature shapes/dims/sizes and capacities...
        """
        return DeepcvModuleDescriptor(self)

    def get_inputs(self) -> Iterable[torch.Tensor]:
        raise NotImplementedError

    def get_outputs(self) -> Iterable[torch.Tensor]:
        raise NotImplementedError

    @property
    def freeze_shared_image_embedding_block(self) -> bool:
        return self._freeze_shared_image_embedding_block

    @property.setter
    def set_freeze_shared_image_embedding_block(self, freeze_weights: bool):
        if self._enable_shared_image_embedding_block:
            self._freeze_shared_image_embedding_block = freeze_weights
            # TODO: freeze/unfreeze weights...
            # TODO: handle concurency between different models training at the same time with unfreezed shared weights
        else:
            logging.warn(self.__class__.SHARED_BLOCK_DISABLED_WARNING_MSG.format('set_freeze_shared_image_embedding_block'))

    def fork_shared_image_embedding_block(self) -> bool:
        """ Copies/forks basic image embedding convolution block's shared weights to be specific to current model (won't be shared anymore)
        Returns whether shared image embedding block have been sucefully forked in current model.
        # TODO: Implementation
        """
        if self._enable_shared_image_embedding_block:
            raise NotImplementedError
            self._shared_block_forked = True
            return True
        else:
            logging.warn(self.__class__.SHARED_BLOCK_DISABLED_WARNING_MSG.format('fork_shared_image_embedding_block'))
        return False

    def merge_shared_image_embedding_block(self):
        """ Merges current model image embedding block's forked weights with shared weights among all DeepCV model
        Won't do anyhthing if image embedding block haven't been forked previously or if they haven't been modified.
        Once image embedding block parameters have been merged with shared ones, current model image embedding block won't be forked anymore (shared weights).
        Returns whether forked image embedding block have been sucefully merged with shared parameters.
        # TODO: Implementation
        """
        if self._enable_shared_image_embedding_block:
            raise NotImplementedError
            self._shared_block_forked = False
            return True
        else:
            logging.warn(self.__class__.SHARED_BLOCK_DISABLED_WARNING_MSG.format('merge_shared_image_embedding_block'))
        return False

    def _define_nn_architecture(self, architecture_spec, submodule_creators: Optional[Dict[str, Callable]] = None, extend_basic_submodule_creators_dict: bool = True):
        """ Defines neural network architecture by parsing 'architecture' hyperparameter and creating sub-modules accordingly
        NOTE: defines `self._features_shapes`, `self._submodules_capacities` and `self._architecture_spec` attributes (usefull for debuging and `self.__str__` and `self.describe` functions)
        Args:
            - architecture_spec: Neural net architecture definition listing submodules to be created with their respective parameters (probably from hyperparameters of `conf/base/parameters.yml` configuration file)
            - submodule_creators: Dict of possible architecture sub-modules associated with their respective module creators. If None, then defaults to `deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS`.
            - extend_basic_submodule_creators_dict: Boolean indicating whether `submodule_creators` argument will be extended with `deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS` dict or not. i.e. whether `submodule_creators` defines additionnal sub-modules or all existing sub-modules. (if `True` and some submodule name(s) (i.e. Dict key(s)) are both present in `submodule_creators` and  `deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS`, then `submodule_creators` dict values (submodule creator(s) Callable(s)) will override defaults/basic one(s)).
        """
        self._features_shapes = [self._input_shape]
        self._architecture_spec = architecture_spec
        self._submodules_capacities = []

        if submodule_creators is None:
            submodule_creators = BASIC_SUBMODULE_CREATORS
        elif extend_basic_submodule_creators_dict:
            submodule_creators = {**BASIC_SUBMODULE_CREATORS, **submodule_creators}

        modules = []
        for i, (name, params) in enumerate(architecture_spec):
            # Create layer/block module from module_creators function dict
            fn = submodule_creators.get(name)
            if not fn:
                # If we can't find suitable function in module_creators, we try to evaluate function name (allows external functions to be used to define model's modules)
                try:
                    fn = utils.get_by_identifier(name)
                except Exception as e:
                    raise RuntimeError(f'Error: Could not locate module/function named "{name}" given module creators: "{submodule_creators.keys()}"') from e
            available_params = {'layer_params': params, 'prev_shapes': self._features_shapes, 'hp': self._hp}
            module = fn(**{n: p for n, p in available_params if n in inspect.signature(fn).parameters})
            modules.append(f'module_{i}', module)
            self._submodules_capacities.append(meta.nn.get_model_capacity(module))

            # Get neural network output features shapes by performing a dummy forward
            with torch.no_grad():
                dummy_batch_x = torch.unsqueeze(torch.zeros(self._input_shape), dim=0)
                self._features_shapes.append(nn.Sequential(*modules)(dummy_batch_x).shape)
        return nn.Sequential(OrderedDict(modules))

    def _initialize_parameters(self, act_fn: Optional[Type[nn.Module]] = None):
        xavier_gain = nn.init.calculate_gain(meta.nn.get_gain_name(act_fn)) if act_fn else None

        def _raise_if_no_act_fn(sub_module_name: str):
            if xavier_gain is None:
                msg = f'Error: Must specify  in `DeepcvModule.initialize_parameters` function in order to initialize {sub_module_name} layer sub-module with xavier initialization.'
                raise RuntimeError(msg)

        def _xavier_init(module: nn.Module):
            if meta.nn.is_conv(module):
                _raise_if_no_act_fn('convolution')
                nn.init.xavier_normal_(module.weight.data, gain=xavier_gain)
                module.bias.data.fill_(0.)
            elif utils.is_fully_connected(module):
                _raise_if_no_act_fn('fully connected')
                nn.init.xavier_uniform_(module.weight.data, gain=xavier_gain)
                module.bias.data.fill_(0.)
            elif type(module).__module__ == nn.BatchNorm2d.__module__:
                nn.init.uniform_(module.weight.data)  # gamma == weight here
                module.bias.data.fill_(0.)  # beta == bias here
            elif list(module.parameters(recurse=False)) and list(module.children()):
                raise Exception("ERROR: Some module(s) which have parameter(s) haven't bee explicitly initialized.")
        self.apply(_xavier_init)

    @classmethod
    def _define_shared_image_embedding_block(cls):
        logging.info('Creating shared image embedding block of DeepcvModule models...')
        raise NotImplementedError
        layers = []
        cls.shared_image_embedding_block = nn.Sequential(OrderedDict(layers))


class DeepcvModuleDescriptor:
    """ Describes DeepCV module with its architecture, capacity and features shapes at sub-modules level """

    def __init__(self, module: DeepcvModule):
        self.module = module

        if '_architecture_spec' in module.__dict__:
            # NOTE: `module.architecture_spec` attribute will be defined if `module._define_nn_architecture` is called
            architecture_spec = module._architecture_spec
        elif 'architecture' in module._hp:
            # otherwise, we try to look for architecture/sub-modules configuration in hyperparameters dict
            architecture_spec = module._hp['architecture']
        else:
            architecture_spec = None
            logging.warn(f"Warning: `{self.__class__.__name__}({module.__class__})`: cant find NN architecture, no `module.architecture_spec` attr. nor `architecture` in `module._hp`")

        # Fills and return a DeepCV module descriptor
        self.capacity = meta.nn.get_model_capacity(module)
        self.human_readable_capacity = utils.human_readable_size(capacity)
        self.model_class = module.__class__
        self.model_class_name = module.__class__.__name__

        if '_features_shapes' in module.__dict__:
            self.submodules_features_shapes = module._features_shapes
            self.submodules_features_dims = map(len, module._features_shapes)
            self.submodules_features_sizes = map(np.prod, module._features_shapes)
        if architecture_spec is not None:
            self.architecture = architecture_spec
            self.submodules_types = [n for n, v in architecture_spec]
        if '_submodules_capacities' in module.__dict__:
            self.submodules_capacities = module._submodules_capacities
            self.human_readable_capacities = map(utils.human_readable_size, module._submodules_capacities)

    def __str__(self) -> str:
        """ Ouput a human-readable string representation of the deepcv module based on its descriptor """
        if self.architecture is not None:
            features_shapes = self._features_shapes if '_features_shapes'
            capas = map(utils.human_readable_size, self._submodules_capacities) if '_submodules_capacities' in self.__dict__ else ['UNKNOWN'] * len(architecture_spec)
            modules_str = '\n\t'.join([f'- {n}({p}) output_features_shape={s}, capacity={c}' for (n, p), s, c in zip(architecture_spec, features_shapes, capas)])
        else:
            modules_str = '(No architecture informations to describe)'
        return f'{self.__class__.__name__} (capacity={capacity_str}):\n\t{modules_str}'


def _create_avg_pooling(layer_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: meta.hyperparams.Hyperparameters) -> nn.Module:
    prev_dim = len(prev_shapes[1:])
    if prev_dim >= 4:
        return nn.AvgPool3d(**layer_params)
    elif prev_dim >= 2:
        return nn.AvgPool2d(**layer_params)
    return nn.AvgPool1d(**layer_params)


def _create_conv2d(layer_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: meta.hyperparams.Hyperparameters) -> nn.Module:
    layer_params['in_channels'] = prev_shapes[-1][1]
    layer = meta.nn.conv_layer(layer_params, hp['act_fn'], hp['dropout_prob'], hp['batch_norm'])
    return layer


def _create_fully_connected(layer_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: meta.hyperparams.Hyperparameters) -> nn.Module:
    layer_params['in_features'] = np.prod(prev_shapes[-1][1:])  # We assume here that features/inputs are given in batches
    if 'out_features' not in layer_params:
        # Handle last fully connected layer (no dropout nor batch normalization for this layer)
        layer_params['out_features'] = self._output_size  # TODO: handle output layer elsewhere
        return meta.nn.fc_layer(layer_params)
    return meta.nn.fc_layer(layer_params, hp['act_fn'], hp['dropout_prob'], hp['batch_norm'])


if __name__ == '__main__':
    test_module(__file__)
