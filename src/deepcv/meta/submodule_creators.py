#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Base submodule creators meta module - submodules_creators.py - `DeepCV`__
Base submodule creators meta module for base_module.DeepcvModule NN definition from YAML specs
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import inspect
from functools import partial
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence, List, Set

import torch
import torch.nn
import torch.nn.functional as F

import numpy as np
import nni
import nni.nas.pytorch.mutables as nni_mutables

import deepcv.utils
import deepcv.meta.nn
from deepcv.meta.nn_spec import yaml_tokens, TENSOR_REDUCTION_FNS
from deepcv.meta.types_aliases import *

__all__ = ['BASIC_SUBMODULE_CREATORS', 'ForwardCallbackSubmodule', 'submodule_creator_dec',
           'avg_pooling_creator', 'new_branch_creator', 'add_nn_layer_creator', 'add_residual_dense_link_creator']
__author__ = 'Paul-Emmanuel Sotir'

#___________________________________________ SUBMODULE CREATORS CONSTANTS _____________________________________________#


""" Default submodule types (defined by a name associated to a submodule creator function) available in 'deepcv.meta.base_module.DeepcvModule' YAML NN architecture specification.  
This list can be extended or overriden according to your needs by providing your own submodule creator functions to `DeepcvModule`'s `__init__()` method ad/or using `deepcv.meta.submodule_creators.submodule_creator_dec`  
NOTE: By default, there are other possible submodules which are builtin DeepcvModule: see `deepcv.meta.nn_spec.yaml_tokens.NESTED_DEEPCV_MODULE`, `deepcv.meta.nn_spec.yaml_tokens.NAS_LAYER_CHOICE` and `deepcv.meta.nn_spec.yaml_tokens.NEW_BRANCH_FROM_TENSOR`  
NOTE: You can add other base submodule creators to this dictionary by using `deepcv.meta.submodule_creators.submodule_creator_dec` function decorator (see example usage in `deepcv.meta.submodule_creators.avg_pooling_creator`)
"""
BASIC_SUBMODULE_CREATORS = {'concat_coords': deepcv.meta.nn.ConcatCoords, 'concat_hilbert_coords': deepcv.meta.nn.ConcatHilbertCoords}


#____________________________________________ SUBMODULE CREATORS CLASSES ______________________________________________#


class ForwardCallbackSubmodule(torch.nn.Module):
    """ `DeepcvModule`-specific Pytorch module which is defined from a callback called on forward passes.
    This Pytorch module behavior is only defined from given callback which makes it more suitable for residual/dense links, for example.
    `ForwardCallbackSubmodule`s are handled in a specific way by `DeepcvModule` for builtin support for output tensor references (e.g. residual links with `deepcv.meta.nn_spec.yaml_tokens.FROM` parameter, see `add_residual_dense_link_creator` for example usage in a submodule creator)
    , reduction functions support (see TENSOR_REDUCTION_FNS for more details) and NNI NAS Mutable InputChoice support. It means that `DeepcvModule` will parse reduction function and output tensor references and provide those in `self.reduction_fn`, `self.mutable_input_choice` before any forward passes and give those along with `referenced_submodules_out` argument to forward pass callback.
    NOTE: `tensor_reduction_fn` and `referenced_submodules_out` arguments are not mandatory in forward callback signature from a submodule creator, but can be taken according to your needs (if reduction function, tensor references and/or NNI NAS Mutable InputChoice support is needed for this NN submodule).
    """

    def __init__(self, forward_callback: SUBMODULE_FORWARD_CALLBACK_T):
        self.forward_callback: SUBMODULE_FORWARD_CALLBACK_T = forward_callback
        forward_callback_signature = inspect.signature(self._forward_callback).parameters
        self.takes_tensor_references: bool = 'referenced_submodules_out' in forward_callback_signature
        self.takes_reduction_fn: bool = 'tensor_reduction_fn' in forward_callback_signature
        # `self.mutable_input_choice` is filled at runtime by `DeepcvModule` when parsing architecture from NN YAML spec. (i.e. wont be `None` during forward passes if an NNI NAS input choice is specified in NN specs.)
        self.mutable_input_choice: nni_mutables.InputChoice = None
        # `self.reduction_fn` is filled at runtime by `DeepcvModule` when parsing architecture from NN YAML spec. (i.e. wont be `None` during forward passes if a reduction function is specified explicitly in NN specs.)
        self.reduction_fn: REDUCTION_FN_T = None
        # `self.referenced_submodules` is filled at runtime by `DeepcvModule` when parsing architecture from NN YAML spec. (i.e. wont be `None` during forward passes if tensor reference(s) are specified in NN specs., e.g. using `deepcv.meta.nn_spec.yaml_tokens.FROM` or `deepcv.meta.nn_spec.yaml_tokens.FROM_NAS_INPUT_CHOICE`)
        self.referenced_submodules: Tuple[str] = None

    def forward(self, x: TENSOR_OR_SEQ_OF_TENSORS_T, referenced_output_features: Optional[Dict[str, TENSOR_OR_SEQ_OF_TENSORS_T]] = None) -> TENSOR_OR_SEQ_OF_TENSORS_T:
        # If needed, give referenced output tensor(s) and a reduction function ('sum', 'mean', 'concat' or 'none' reduction) arguments to forward callback
        forward_kwargs = {}
        if referenced_output_features is not None and self.referenced_submodules is not None and self.takes_tensor_references:
            referenced_output_features = [v for n, v in referenced_output_features.items() if n in self.referenced_submodules]
            forward_kwargs['referenced_submodules_out'] = referenced_output_features if self.mutable_input_choice is None else self.mutable_input_choice(referenced_output_features)
        if self.tensor_reduction_fn is not None and self.takes_reduction_fn:
            forward_kwargs['tensor_reduction_fn'] = self.tensor_reduction_fn

        # Forward pass through sub-module based on forward pass callback
        return self.forward_callback(x, **forward_kwargs)


#___________________________________________ SUBMODULE CREATORS FUNCTIONS _____________________________________________#


def submodule_creator_dec(name: str, submodule_creators: SUBMODULE_CREATORS_DICT_T = BASIC_SUBMODULE_CREATORS) -> Callable[[Callable], Callable]:
    """ Decorator helper function which appends a new entry to `submodule_creators` with decorated function associated to its `name`. """
    assert name not in submodule_creators, f'Error: "{name}" submodule creator entry already exists, can have duplicate submodule creator names.'

    def _decorator(creator: Callable[..., torch.nn.Module]):
        submodule_creators[name] = creator
        return creator
    return _decorator


@submodule_creator_dec(name='average_pooling')
def avg_pooling_creator(submodule_params: Dict[str, Any], prev_shapes: List[torch.Size]) -> torch.nn.Module:
    prev_dim = len(prev_shapes[-1][1:])
    if prev_dim >= 4:
        return torch.nn.AvgPool3d(**submodule_params)
    elif prev_dim >= 2:
        return torch.nn.AvgPool2d(**submodule_params)
    return torch.nn.AvgPool1d(**submodule_params)


@submodule_creator_dec(name=yaml_tokens.NEW_BRANCH_FROM_TENSOR)
def new_branch_creator(submodule_params: Dict[str, Any], channel_dim: int = 1) -> ForwardCallbackSubmodule:
    if yaml_tokens.FROM not in submodule_params and yaml_tokens.FROM_NAS_INPUT_CHOICE not in submodule_params:
        raise ValueError(f'Error: You must either specify "{yaml_tokens.FROM}" or "{yaml_tokens.FROM_NAS_INPUT_CHOICE}" parameter '
                         f'in a "{yaml_tokens.NEW_BRANCH_FROM_TENSOR}" submodule spec.')

    def _new_branch_from_tensor_forward(x: TENSOR_OR_SEQ_OF_TENSORS_T, referenced_submodules_out: List[torch.Tensor], tensor_reduction_fn: REDUCTION_FN_T = TENSOR_REDUCTION_FNS['concat'], channel_dim: int = 1):
        """ Simple forward pass callback which takes referenced output tensor(s) and ignores previous submodule output features, allowing to define siamese/parallel branches thereafter.
        In other words, `deepcv.meta.nn_spec.yaml_tokens.NEW_BRANCH_FROM_TENSOR` submodules are similar to dense links but will only use referenced submodule(s) output, allowing new siamese/parrallel NN branches to be defined (wont reuse previous submodule output features)
        If multiple tensors are referenced using `deepcv.meta.nn_spec.yaml_tokens.FROM` (or `deepcv.meta.nn_spec.yaml_tokens.FROM_NAS_INPUT_CHOICE`), `tensor_reduction_fn` reduction function will be applied.
        Reduction function is 'concat' by default and can be overriden by `_reduction` parameter in link submodule spec., see `deepcv.meta.nn_spec.TENSOR_REDUCTION_FNS` for all possible reduction functions.
        """
        # Ignores `x` input tensor (previous submodule output tensor is ignored)
        return tensor_reduction_fn(referenced_submodules_out, dim=channel_dim)
    return ForwardCallbackSubmodule(partial(_new_branch_from_tensor_forward, channel_dim=channel_dim))


def add_nn_layer_creator(layer_op_t: Type[torch.nn.Module], creator_name: str, submodule_creators: Dict[str, Callable] = BASIC_SUBMODULE_CREATORS) -> Callable[['submodule_params', 'prev_shapes', 'act_fn', 'dropout_prob', 'batch_norm', 'channel_dim', 'preactivation'], torch.nn.Module]:
    """ Creates a fully connected or convolutional NN layer with optional dropout and batch/layer/instance/group norm support (and preactivation, activation function, ... choices)
    NOTE: We assume here that features/inputs are given in (mini)batches (`channel_dim` defaults to 1)
    """
    if deepcv.meta.nn.is_conv(layer_op_t) and not isinstance(layer_op_t, torch.nn.Linear):
        raise TypeError(f'Error: Wrong `layer_op_t` type, cant create a NN layer of type {layer_op_t} with `deepcv.meta.submodule_creators.add_nn_layer_creator` '
                        'submodule creator (`layer_op_t` should either be a convolution or a `torch.nn.Linear`).')

    @submodule_creator_dec(name=creator_name, submodule_creators=submodule_creators)
    def _nn_layer_creator(submodule_params: Dict[str, Any], prev_shapes: List[torch.Size], layer_op_t: Type[torch.nn.Module], act_fn: Optional[torch.nn.Module] = None, dropout_prob: Optional[float] = None, channel_dim: int = 1, preactivation: bool = False,
                          batch_norm: Optional[Dict[str, Any]] = None, layer_norm: Optional[Dict[str, Any]] = None, instance_norm: Optional[Dict[str, Any]] = None, group_norm: Optional[Dict[str, Any]] = None, layer_norm_with_mean_only_batch_norm: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        # Handle specified normalization techniques if any (BatchNorm, LayerNorm, InstanceNorm, GroupNorm and/or layer_norm_with_mean_only_batch_norm)
        norm_techniques = {deepcv.meta.nn.NormTechnique.BATCH_NORM: batch_norm, deepcv.meta.nn.NormTechnique.LAYER_NORM: layer_norm, deepcv.meta.nn.NormTechnique.INSTANCE_NORM: instance_norm,
                           deepcv.meta.nn.NormTechnique.GROUP_NORM: group_norm, deepcv.meta.nn.NormTechnique.LAYER_NORM_WITH_MEAN_ONLY_BATCH_NORM: layer_norm_with_mean_only_batch_norm}
        norms = {t: args for t, args in norm_techniques.items() if args is not None and len(args) > 0}

        # Only supports convolutions and linear layers in this submodule creator
        if isinstance(layer_op_t, torch.nn.Linear) and 'in_features' not in submodule_params:
            submodule_params['in_features'] = np.prod(prev_shapes[-1][channel_dim:])
        elif deepcv.meta.nn.is_conv(layer_op_t) and 'in_channels' not in submodule_params:
            submodule_params['in_channels'] = prev_shapes[-1][channel_dim]

        return deepcv.meta.nn.layer(layer_op=layer_op_t(**submodule_params), act_fn=act_fn, dropout_prob=dropout_prob, preactivation=preactivation,
                                    norm_type=norms.keys(), norm_kwargs=norms.values(), input_shape=prev_shapes[-1][channel_dim:])

    _nn_layer_creator.__doc__ = add_nn_layer_creator.__doc__
    return partial(_nn_layer_creator, layer_op_t=layer_op_t)


# Add NN Layer submodule creator entries to `BASIC_SUBMODULE_CREATORS`
add_nn_layer_creator(layer_op_t=torch.nn.Conv1d, creator_name='conv1d')
add_nn_layer_creator(layer_op_t=torch.nn.Conv2d, creator_name='conv2d')
add_nn_layer_creator(layer_op_t=torch.nn.Conv3d, creator_name='conv3d')
add_nn_layer_creator(layer_op_t=torch.nn.ConvTranspose1d, creator_name='transosed_conv1d')
add_nn_layer_creator(layer_op_t=torch.nn.ConvTranspose2d, creator_name='transosed_conv2d')
add_nn_layer_creator(layer_op_t=torch.nn.ConvTranspose3d, creator_name='transosed_conv3d')
add_nn_layer_creator(layer_op_t=torch.nn.Linear, creator_name='linear')
add_nn_layer_creator(layer_op_t=torch.nn.Linear, creator_name='fully_connected')


def add_residual_dense_link_creator(is_residual: bool, creator_name: str, submodule_creators: Dict[str, Callable] = BASIC_SUBMODULE_CREATORS) -> Callable[['submodule_params', 'allow_scaling', 'scaling_mode', 'channel_dim'], ForwardCallbackSubmodule]:
    """ Creates a residual or dense link sub-module which concatenates or adds features from previous sub-module output with other referenced sub-module(s) output(s).
    `submodule_params` argument must contain a `deepcv.meta.nn_spec.yaml_tokens.FROM` (or `deepcv.meta.nn_spec.yaml_tokens.FROM_NAS_INPUT_CHOICE`) entry giving the other sub-module reference(s) (sub-module name(s)) from which output(s) is added to previous sub-module output features.

    Returns a callback which is called during foward pass of DeepcvModule.
    Like any other DeepcvModule submodule creators which returns a forward callback and which uses tensor references (`deepcv.meta.nn_spec.yaml_tokens.FROM` or `deepcv.meta.nn_spec.yaml_tokens.FROM_NAS_INPUT_CHOICE`), a reduction function can be specified (see `REDUCTION_FN_T`) in `deepcv.meta.nn_spec.yaml_tokens.FROM` or `deepcv.meta.nn_spec.yaml_tokens.FROM_NAS_INPUT_CHOICE` parameters: by default, reduction function will be 'sum' if this is a residual link and 'concat' if this is a dense link.
    If 'allow_scaling' is `True` (when `allow_scaling: True` is specified in YAML residual/dense link spec.), then if residual/dense features have different shapes on dimensions following `channel_dim`, it will be scaled (upscaled or downscaled) using [`torch.functional.interpolate`](https://pytorch.org/docs/stable/nn.functional.html#interpolate).
    By default interpolation is 'linear'; If needed, you can specify a `scaling_mode` parameter as well to change algorithm used for up/downsampling (valid values: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area', for more details, see [`torch.nn.functional.interpolate` doc](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate)).
    NOTE: Scaling/interpolation of residual/dense tensors is only supported for 1D, 2D and 3D features, without taking into account channel and minibatch dimensions. Also note that `minibatch` dimension is required by [`torch.functional.interpolate`](https://pytorch.org/docs/stable/nn.functional.html#interpolate).
    NOTE: If `allow_scaling` is `False`, output features shapes of these two or more submodules must be the same, except for the channels/filters dimension if this is a dense link.
    NOTE: The only diference between residual and dense links ('is_residual' beeing 'True' of 'False') is the default `deepcv.meta.nn_spec.yaml_tokens.REDUCTION_FN` function beeing respectively 'sum' and 'concat'.
    """
    @submodule_creator_dec(name=creator_name, submodule_creators=submodule_creators)
    def _link_creator(submodule_params: Dict[str, Any], is_residual: bool, allow_scaling: bool = False, scaling_mode: str = 'linear', channel_dim: int = 1) -> ForwardCallbackSubmodule:
        if yaml_tokens.FROM not in submodule_params and yaml_tokens.FROM_NAS_INPUT_CHOICE not in submodule_params:
            raise ValueError(f'Error: Missing "{yaml_tokens.FROM}" or "{yaml_tokens.FROM_NAS_INPUT_CHOICE}" parameter in '
                             f'{"residual" if is_residual else "dense"} link YAML specification; You should at least provide a tensor reference.')

        def _forward_callback(x: TENSOR_OR_SEQ_OF_TENSORS_T, referenced_submodules_out: List[torch.Tensor], tensor_reduction_fn: REDUCTION_FN_T = (TENSOR_REDUCTION_FNS['sum'] if is_residual else TENSOR_REDUCTION_FNS['concat'])) -> TENSOR_OR_SEQ_OF_TENSORS_T:
            """ Redisual or Dense link forward pass callbacks
            If target output shape is different from one of the referenced tensor shapes, a up/down-scaling (interpolation) may be performed according to `scaling_mode` and `allow_scaling` parameters.
            NOTE: Target output shape is assumed to be the same as `x` features shape if `x` is a `torch.Tensor` or the same as the first tensor shape of `x` if `x` is a list of `torch.Tensor`s.
            A reduction function can be specified (see `deepcv.meta.nn_spec.TENSOR_REDUCTION_FNS`); If this is a residual link, reduction function defaults to 'sum' and if this is a dense link, reduction function defaults to 'concat'.
            """
            tensors = [x, ] if isinstance(x, torch.Tensor) else [*x, ]
            for y in referenced_submodules_out:
                if tensors[0].shape[channel_dim + 1:] != y.shape[channel_dim + 1:]:
                    if allow_scaling:
                        # Resize y features tensor to be of the same shape as x along dimensions after channel dim (scaling performed with `torch.nn.functional.interpolate`)
                        y = F.interpolate(y, size=x.shape[channel_dim + 1:], mode=scaling_mode)
                    else:
                        raise RuntimeError(f"Error: Couldn't forward throught {'residual' if is_residual else 'dense'} link: features from link doesn't have "
                                           f"the same shape as previous module's output shape, can't concatenate or add them. (did you forgot to allow residual/dense "
                                           f"features to be scaled using `allow_scaling: true` parameter?). `residual_shape='{y.shape}' != prev_features_shape='{x.shape}'`")
                tensors.append(y)

            # Add or concatenate previous sub-module output features with residual or dense features
            return tensor_reduction_fn(tensors, dim=channel_dim)
        return ForwardCallbackSubmodule(_forward_callback)

    _link_creator.__doc__ = add_residual_dense_link_creator.__doc__
    return partial(_link_creator, is_residual=is_residual)


# Add Residual and Dense Link submodule creator entries to `BASIC_SUBMODULE_CREATORS`
add_residual_dense_link_creator(is_residual=True, creator_name='residual_link')
add_residual_dense_link_creator(is_residual=False, creator_name='dense_link')


#___________________________________________ SUBMODULE CREATORS UNIT TESTS ____________________________________________#

if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
