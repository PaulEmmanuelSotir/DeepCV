#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" DeepCV model base class meta module - base_module.py - `DeepCV`__
Defines DeepCV model base class
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import inspect
import logging
from collections import OrderedDict
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence

import torch
import torch.nn as nn

import deepcv.meta as meta
import deepcv.utils as utils
from tests.tests_utils import test_module

__all__ = ['DeepcvModule']
__author__ = 'Paul-Emmanuel Sotir'


class DeepcvModule(nn.Module):
    """ DeepCV PyTorch Module model base class
    Handles hyperparameter defaults, NN architecture definition tooling and basic shared convolution block for transfert learning between all DeepCV models
    Child class must define `HP_DEFAULTS` class attribute, with at least the following keys: `{'architecture': ..., 'act_fn': ...}` and other needed hyperparameters deepending on which sub-module are specified in `architecture` definition
    For more details about `architecture` hyperparameter parsing, see code in `DeepcvModule._define_nn_architecture`.
    # TODO: implement basic conv block shared by all DeepcvModules (frozen weights by default, and allow forking of these weights to be specific to a given model)
    # TODO: move code from ObjectDetector into DeepcvModule
    """

    HP_DEFAULTS = ...
    SHARED_BLOCK_DISABLED_WARNING_MSG = r'Warning: `DeepcvModule.{}` called while `self._enable_shared_image_embedding_block` is `False` (Shared image embedding block disabled for this model)'

    def __init__(self, input_shape: torch.Size, hp: meta.hyperparams.Hyperparameters, enable_shared_block: bool = True, freeze_shared_block: bool = True):
        super(self.__class__).__init__(self)

        # Process module hyperparameters
        assert self.__class__.HP_DEFAULTS != ..., f'Error: Module classes which inherits from "DeepcvModule" ({self.__class__.__name__}) must define "HP_DEFAULTS" class attribute dict.'
        self.HP_DEFAULTS.update(_BASE_DEECV_MODULE_DEFAULTS)
        self._hp, missing_hyperparams = hp.with_defaults(self.__class__.HP_DEFAULTS)
        assert len(missing_hyperparams) > 0, f'Error: Missing required hyper-parameter in "{self.__class__.__name__}" module parameters. (missing: "{missing_hyperparams}")'

        self._input_shape = input_shape
        self._shared_block_forked = False
        self._features_shapes = [self.input_shape]
        self._enable_shared_image_embedding_block = enable_shared_block

        self.freeze_shared_image_embedding_block = freeze_shared_block
        if enable_shared_block and not 'shared_image_embedding_block' in self.__class__.__dict__:
            # If class haven't been instanciated yet, define common/shared DeepcvModule image embedding block
            self.__class__._define_shared_image_embedding_block()

    def __str__(self) -> str:
        capacity = utils.human_readable_size(meta.nn.get_model_capacity(self))
        modules_str = '\n\t'.join([f'- {n}({p}) output_features_shape={s}' for (n, p), s in zip(self._hp['architecture'], self._features_shapes)])
        return f'{self.__class__.__name__} (capacity={capacity}):\n\t{modules_str}'

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

    def _define_nn_architecture(self, architecture_spec, submodule_creators: Dict[str, Callable]):
        """ Defines neural network architecture by parsing 'architecture' hyperparameter and creating sub-modules accordingly """
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
            modules.append((f'module_{i}', fn(**{n: p for n, p in available_params if n in inspect.signature(fn).parameters})))

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


if __name__ == '__main__':
    test_module(__file__)
