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

BASIC_SUBMODULE_CREATORS = {'avg_pooling': _create_avg_pooling, 'conv2d': _create_conv2d, 'fully_connected': _create_fully_connected,
                            'residual_link': _residual_dense_link(is_residual=True), 'dense_link': _residual_dense_link(is_residual=False)}
MODULE_CREATOR_CALLBACK_RETURN_T = Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]
MODULE_CREATOR_MODULE_RETURN_T = nn.Module


class DeepcvModule(nn.Module):
    """ DeepCV PyTorch Module model base class
    Handles hyperparameter defaults, NN architecture definition tooling and basic shared convolution block for transfert learning between all DeepCV models
    Child class must define `HP_DEFAULTS` class attribute, with at least the following keys: `{'architecture': ..., 'act_fn': ...}` and other needed hyperparameters deepending on which sub-module are specified in `architecture` definition
    For more details about `architecture` hyperparameter parsing, see code in `DeepcvModule._define_nn_architecture`.
    NOTE: in order `_features_shapes`, `_submodules_capacities` and `self._architecture_spec` attributes to be defined and contain NN sumbmodules informations, you need to call `DeepcvModule._define_nn_architecture` or update it by yourslef according to your NN architecture.
    NOTE: `self.__str__` outputs a human readable string describing NN's architecture with their respective feature_shape and capacity. In order to be accurate, you need to call `self._define_nn_architecture` or, alternatively, keep `_features_shapes` and `_submodules_capacities` attribute up-to-date and make sure that `self._architecture_spec` or self._hp['architecture'] contains architecture definition (similar value than `self._define_nn_architecture`'s `architecture_spec` argument would have).
    NOTE: A sub-module's name defaults to 'submodule_{i}' where 'i' is sub-module index in architecture sub-module list. Alternatively, you can specify a sub-module's name in architecture configuration, which is preferable for residual/dense links for example.
    .. See examples of Deepcv model sub-modules architecture definition in `[Kedro hyperparameters YAML config file]conf/base/parameters.yml`
    # TODO: implement basic conv block shared by all DeepcvModules (shallow convs with zero-padding, output_shape must be same as input shape except for channel dim) (frozen weights by default, and allow forking of these weights to be specific to a given model) + update architecture_spec/features_shapes
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

    def forward(self, intput_img: torch.Tensor, channel_dim: int = 3):
        # Apply shared image embedding block and combine it's output with input image (concats features over channel dimension)
        x = torch.cat([intput_img, self.shared_image_embedding_block(intput_img)], dim=channel_dim)

        # Apply object detector neural net architecture on top of shared image embedding features and input image
        if len(self._forward_callbacks) > 0:
            # TODO: find a better mechanism to avoid storage of all submodules output features during forward pass in case of residual/dense links (without too much added complexity)
            prev_features_tensors = {}
            for name, subm in self._submodules:
                if name in self._forward_callbacks:
                    x = self._forward_callbacks(x, prev_features_tensors)
                else:
                    x = subm(x)
                prev_features_tensors[name] = x
                return x
        else:
            return self._net(x)

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
        NOTE: defines `self._features_shapes`, `self._submodules_capacities`, `self._forward_callbacks`, `self._submodules` and `self._architecture_spec` attributes (usefull for debuging and `self.__str__` and `self.describe` functions)
        Args:
            - architecture_spec: Neural net architecture definition listing submodules to be created with their respective parameters (probably from hyperparameters of `conf/base/parameters.yml` configuration file)
            - submodule_creators: Dict of possible architecture sub-modules associated with their respective module creators. If None, then defaults to `deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS`.
            - extend_basic_submodule_creators_dict: Boolean indicating whether `submodule_creators` argument will be extended with `deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS` dict or not. i.e. whether `submodule_creators` defines additionnal sub-modules or all existing sub-modules. (if `True` and some submodule name(s) (i.e. Dict key(s)) are both present in `submodule_creators` and  `deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS`, then `submodule_creators` dict values (submodule creator(s) Callable(s)) will override defaults/basic one(s)).
        """
        self._features_shapes = [self._input_shape]
        self._architecture_spec = architecture_spec
        self._submodules_capacities = []
        self._forward_callbacks = {}
        self._submodules = {}

        if submodule_creators is None:
            submodule_creators = BASIC_SUBMODULE_CREATORS
        elif extend_basic_submodule_creators_dict:
            submodule_creators = {**BASIC_SUBMODULE_CREATORS, **submodule_creators}

        submodules = []
        for i, (submodule_type, params) in enumerate(architecture_spec):
            submodule_name = f'submodule_{i}'
            if issubclass(params, List) or issubclass(params, Tuple):
                # Architecture definition specifies a sub-module name explicitly
                submodule_name, params = *params
            elif issubclass(params, str):
                # Architecture definition specifies a sub-module name explicitly without any other sub-module parameters
                submodule_name = params
                params = {}
            elif not issubclass(params, Dict):
                raise RuntimeError(f'Error: Architecture sub-module spec. must either be a parameters Dict, or a submodule name along with parameters Dict, but got: "{params}".')

            # Try to find sub-module creator or nn.Module which matches `submodule_type`
            fn = submodule_creators.get(submodule_type)
            if not fn:
                # If we can't find suitable function in module_creators, we try to evaluate function name (allows external functions to be used to define model's modules)
                try:
                    fn = utils.get_by_identifier(submodule_type)
                except Exception as e:
                    raise RuntimeError(f'Error: Could not locate module/function named "{submodule_type}" given module creators: "{submodule_creators.keys()}"') from e

            # Create layer/block submodule from its module_creator or its nn.Module.__init__ function (fn)
            available_params = {'submodule_params': params, 'prev_shapes': self._features_shapes, 'hp': self._hp}
            module_or_callback = fn(**{n: p for n, p in available_params if n in inspect.signature(fn).parameters})

            # Figure out fn's output (module creators can return a nn.Module or a callback which is called during forwarding of sub-modules (see self.forward for more details about these callbacks)
            if issubclass(module_or_callback, MODULE_CREATOR_CALLBACK_RETURN_T):
                self._forward_callbacks[submodule_name] = module_or_callback
                submodules.append((submodule_name, None))
            elif issubclass(module_or_callback, nn.Module):
                submodules.append((submodule_name, module_or_callback))
            else:
                msg = f'Error: Wrong sub-module creator function/class __init__ return type (must either be a nn.Module or a `forward` callback of type: `{MODULE_CREATOR_CALLBACK_RETURN_T}`.'
                raise RuntimeError(msg)

                # Get neural network submodules capacity and output features shapes
                self._submodules_capacities.append(meta.nn.get_model_capacity(module))
                self._features_shapes.append(meta.nn.get_out_features_shape(nn.Sequential(*submodules)))

        self._net = nn.Sequential(OrderedDict([(n, m) for (n, m) in submodules if m is not None]))
        self._submodules = dict(*submodules)

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
    def _define_shared_image_embedding_block(cls, in_channels: int = 3):
        logging.info('Creating shared image embedding block of DeepcvModule models...')
        conv_opts = {'act_fn': nn.ReLU, 'batch_norm': {'affine': True, 'eps': 1e-05, 'momentum': 0.0736}}
        layers = [meta.nn.conv_layer(conv2d={'in_channels': in_channels, 'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1}, **conv_opts),
                  meta.nn.conv_layer(conv2d={'in_channels': 8, 'out_channels': 16, 'kernel_size': (3, 3), 'padding': 1}, **conv_opts),
                  meta.nn.conv_layer(conv2d={'in_channels': 16, 'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1}, **conv_opts),
                  meta.nn.conv_layer(conv2d={'in_channels': 8, 'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1}, **conv_opts)]
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
            features = self.submodules_features_shapes if 'submodules_features_shapes' in self.__dict__ else ['UNKNOWN'] * len(self.architecture)
            capas = self.human_readable_capacities if 'human_readable_capacities' in self.__dict__ else ['UNKNOWN'] * len(self.architecture)
            modules_str = '\n\t'.join([f'- {n}({p}) output_features_shape={s}, capacity={c}' for (n, p), s, c in zip(self.architecture, features, capas)])
        else:
            modules_str = '(No architecture informations to describe)'
        return f'{self.model_class_name} (capacity={self.human_readable_capacity}):\n\t{modules_str}'


def _create_avg_pooling(submodule_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: meta.hyperparams.Hyperparameters) -> nn.Module:
    prev_dim = len(prev_shapes[1:])
    if prev_dim >= 4:
        return nn.AvgPool3d(**submodule_params)
    elif prev_dim >= 2:
        return nn.AvgPool2d(**submodule_params)
    return nn.AvgPool1d(**submodule_params)


def _create_conv2d(submodule_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: meta.hyperparams.Hyperparameters, channel_dim: int = -3) -> nn.Module:
    """ Creates a convolutional NN layer with dropout and batch norm support
    NOTE: We assume here that features/inputs are given in batches and that input only comes from previous sub-module (e.g. no direct residual/dense link)
    """
    submodule_params['in_channels'] = prev_shapes[-1][channel_dim]
    return meta.nn.conv_layer(submodule_params, hp['act_fn'], hp['dropout_prob'], hp['batch_norm'])


def _create_fully_connected(submodule_params: Dict[str, Any], prev_shapes: List[torch.Size], hp: meta.hyperparams.Hyperparameters) -> nn.Module:
    """ Creates a fully connected NN layer with dropout and batch norm support
    NOTE: We assume here that features/inputs are given in batches and that input only comes from previous sub-module (e.g. no direct residual/dense link)
    """
    submodule_params['in_features'] = np.prod(prev_shapes[-1][1:])
    return meta.nn.fc_layer(submodule_params, hp['act_fn'], hp['dropout_prob'], hp['batch_norm'])


def _residual_dense_link(is_residual: bool = True):
    """ Creates a residual or dense link sub-module which concatenates or adds features from direct previous sub-module and another sub-module
    Output features shapes of these two submodules must be the same, except for the channels/filters dimension if this is a dense link.
    `submodule_params` Argument must contain a `from` entry giving the other sub-module reference (sub-module name) from which output is added to previous sub-module output.
    Returns a callback which is called during forwarding of sub-modules.
    """
    def _create_link_submodule(submodule_params, prev_named_submodules: Dict[str, nn.Module], prev_submodules: List[nn.Module], channel_dim: int = -3) -> MODULE_CREATOR_CALLBACK_RETURN_T:
        def _forward_callback(x: torch.Tensor, prev_named_submodules_out: Dict[str, torch.Tensor]):
            if submodule_params['from'] not in prev_named_submodules:
                raise ValueError(f"Error: Couldn't find previous sub-module name reference '{submodule_params['from']}' in NN architecture.")
            y = prev_named_submodules[submodule_params['from']]
            return x + y if is_residual else torch.cat([x, y], dim=channel_dim)
        return _forward_callback


if __name__ == '__main__':
    test_module(__file__)
