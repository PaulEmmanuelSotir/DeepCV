#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" DeepCV model base class meta module - base_module.py - `DeepCV`__
Defines DeepCV model base class
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import logging
from collections import OrderedDict
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence

import torch
import torch.nn as nn

__all__ = ['DeepcvModule']
__author__ = 'Paul-Emmanuel Sotir'


class DeepcvModule(nn.Module):
    """ DeepCV PyTorch Module model base class
    Handles hyperparameter defaults and basic shared convolution block for transfert learning between all DeepCV models
    # TODO: implement basic conv block shared by all DeepcvModules (frozen weights by default, and allow forking of these weights to be specific to a given model)
    # TODO: move code from ObjectDetector into DeepcvModule
    """

    HP_DEFAULTS = ...
    SHARED_BLOCK_DISABLED_WARNING_MSG = r'Warning: `DeepcvModule.{}` called while `self._enable_shared_image_embedding_block` is `False` (Shared image embedding block disabled for this model)'

    def __init__(self, input_shape: torch.Size, hp: Dict[str, Any], enable_shared_block: bool = True, freeze_shared_block: bool = True):
        super(DeepcvModule, self).__init__()
        assert self.__class__.HP_DEFAULTS != ..., f'Error: Module classes which inherits from "DeepcvModule" ({self.__class__.__name__}) must define "HP_DEFAULTS" class attribute dict.'

        self._input_shape = input_shape
        self._hp = {n: v for n, v in hp if n in self.__class__.HP_DEFAULTS}
        self._hp.update({n: v for n, v in self.__class__.HP_DEFAULTS if n not in hp and v != ...})
        missing_hyperparams = [n for n in self.__class__.HP_DEFAULTS if n not in self.hyper_params]

        self._forked = False
        self._enable_shared_image_embedding_block = enable_shared_block
        self.freeze_shared_image_embedding_block = freeze_shared_block
        if enable_shared_block and not 'shared_image_embedding_block' in self.__class__.__dict__:
            # If class haven't been instanciated yet, define common/shared DeepcvModule image embedding block
            self.__class__._define_shared_image_embedding_block()

        assert len(missing_hyperparams) > 0, f'Error: Missing required hyper-parameter in "{self.__class__.__name__}" module parameters'

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
            self._forked = True
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
            self._forked = False
            return True
        else:
            logging.warn(self.__class__.SHARED_BLOCK_DISABLED_WARNING_MSG.format('merge_shared_image_embedding_block'))
        return False

    @classmethod
    def _define_shared_image_embedding_block(cls):
        logging.info('Creating shared image embedding block of DeepcvModule models...')
        raise NotImplementedError
        layers = []
        cls.shared_image_embedding_block = nn.Sequential(OrderedDict(layers))
