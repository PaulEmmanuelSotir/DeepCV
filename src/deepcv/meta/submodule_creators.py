#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Base submodule creators meta module - submodules_creators.py - `DeepCV`__
Base submodule creators meta module for base_module.DeepcvModule NN definition from YAML specs
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import inspect
from functools import partial
from collections import OrderedDict
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence, List, Set

import torch
import torch.nn
import torch.nn.functional as F

import numpy as np
import nni
import nni.nas.pytorch.mutables as nni_mutables

from deepcv.utils import parse_slice, import_tests, NL
from .types_aliases import *
from . import nn as deepcv_nn
from .nn_spec import yaml_tokens
from . import hrnet

__all__ = ['BASIC_SUBMODULE_CREATORS', 'get_reduction_fn', 'Reduce', 'TENSOR_REDUCTION_FNS', 'ForwardCallbackSubmodule', 'submodule_creator_dec',
           'avg_pooling_creator', 'reduction_subm_creator', 'select_tensor_creator', 'new_branch_creator', 'add_nn_layer_creator', 'add_residual_dense_link_creator']
__author__ = 'Paul-Emmanuel Sotir'

#________________________________________ CONSTANTS and Reduction function impl _______________________________________#


""" Default submodule types (defined by a name associated to a submodule creator function) available in 'deepcv.meta.base_module.DeepcvModule' YAML NN architecture specification.
This list can be extended or overriden according to your needs by providing your own submodule creator functions to `DeepcvModule`'s `__init__()` method ad/or using `deepcv.meta.submodule_creators.submodule_creator_dec`
NOTE: By default, there are other possible submodules which are builtin DeepcvModule: e.g. `deepcv.meta.nn_spec.yaml_tokens.NESTED_DEEPCV_MODULE`, `deepcv.meta.nn_spec.yaml_tokens.NAS_LAYER_CHOICE` and `deepcv.meta.nn_spec.yaml_tokens.NEW_BRANCH_FROM_TENSOR`. See `deepcv.meta.nn_spec.yaml_tokens` for a more exhaustive list of `DeepcvModule`'s builtin tokens (which includes builtin submodule names).
NOTE: You can add other base submodule creators to this dictionary by using `deepcv.meta.submodule_creators.submodule_creator_dec` function decorator (see example usage in `deepcv.meta.submodule_creators.avg_pooling_creator`)
"""
BASIC_SUBMODULE_CREATORS = {'flatten': deepcv_nn.Flatten, 'concat_coords': deepcv_nn.ConcatCoords, 'concat_hilbert_coords': deepcv_nn.ConcatHilbertCoords,
                            'multiresolution_fusion': hrnet.MultiresolutionFusion, 'parallel_conv': hrnet.ParallelConvolution, 'hrnet_input_stem': hrnet.hrnet_input_stem,
                            'hrnet_repr_head_v1': hrnet.HRNetV1RepresentationHead, 'hrnet_repr_head_vZ': hrnet.HRNetV2RepresentationHead, 'hrnet_repr_head_v2p': hrnet.HRNetV2pRepresentationHead, }  # ... + other creators decorated with `deepcv.meta.submodule_creators.submodule_creator_dec`


def get_reduction_fn(reduction: str) -> Callable[[TENSOR_OR_SEQ_OF_TENSORS_T, ...], TENSOR_OR_SEQ_OF_TENSORS_T]:
    """ Reduction functions implementation. Supported `reduction` values are 'mean', 'sum', 'concat' and 'none'
    NOTE: `keep_dim` argument only applies to 'sum' and 'mean' reduction functions (see 'keep_dim' argument of `torch.sum` and/or `torch.mean`) 
    NOTE: Be aware that `out` argument will be unchanged if 'none' reduction function is applied (can't set `out` from input `tensors` which may contain multiple tensors)
    """
    def _reduction_fn(tensors: TENSOR_OR_SEQ_OF_TENSORS_T, keep_dim: bool = False, out: Optional[torch.Tensor] = None) -> TENSOR_OR_SEQ_OF_TENSORS_T:
        if reduction == 'none':
            return tensors
        if deepcv_nn.is_torch_obj(tensors):
            if out is not None:
                tensors.copy(out)
            return tensors

        tensors = torch.cat(tensors, dim=1, out=out if reduction == 'concat' else None)

        if reduction == 'mean':
            return torch.mean(tensors, dim=1, keep_dim=keep_dim, out=out)
        elif reduction == 'sum':
            return torch.sum(tensors, dim=1, keep_dim=keep_dim, out=out)
        else:
            raise ValueError(f'Error: Invalid "{reduction}" reduction function name. Valid reduction functions are "mean", "sum", "concat" and "none".')
        return tensors
    return _reduction_fn


""" Simple `torch.nn.Module` for standalone reduction function usage. See `deepcv.meta.submodule_creators.TENSOR_REDUCTION_FNS` and `deepcv.meta.submodule_creatorsget_reduction_fn` for more details.
# TODO: decide whether to keep this implementation with func_to_module and/or `reduction_subm_creator` implementation (ideally, refactor code in order to remove ForwardCallaback submodule (redoundant with and too complex/sketchy: move its logic to `submodule_creator_dec` and replace some of its usage with `func_to_module` usage))
"""
Reduce = deepcv_nn.func_to_module('Reduce', init_params=['reduction', 'keep_dim'])(
    lambda tensors, reduction, keep_dim, out: get_reduction_fn(reduction)(tensors, keep_dim=keep_dim, out=out))


""" Available/supported reduction functions. These can either be used in `reduce` submodules or be specified in various submodules spec (`reduction` spec. parameter).
NOTE: A reduction function parameter is also used in YAML specification of NNI NAS Mutable Layer Choice submodules (`yaml_tokens.REDUCTION_FN` param), but, unlike other submodules, NNI NAS `LayerChoice`(s) uses NNI builtin reduction function implementation instead of DeepCV's `TENSOR_REDUCTION_FNS`/`get_reduction_fn`; However, both supports the same reduction function names.
NOTE: If you extend `TENSOR_REDUCTION_FNS` with your own reduction functions, make sure they take `dim`, `keep_dim` and `out` args along with input tensors. All reduction function should have the same signature in order to simplify their usage (ignore unapplicable/unused args); Otherwise, may raise exceptions in 'deepcv.meta.base_module.DeepcvModule', 'deepcv.meta.submodule_creators' and/or 'deepcv.meta.nn_spec'.
"""
TENSOR_REDUCTION_FNS = {reduction: get_reduction_fn(reduction) for reduction in ['mean', 'sum', 'concat', 'none']}


#____________________________________________ FORWARD CALLBACK HELPER CLASS ______________________________________________#


class ForwardCallbackSubmodule(torch.nn.Module):
    """ `DeepcvModule`-specific Pytorch module which is defined from a callback called on forward passes.
    This Pytorch module behavior is only defined from given callback which makes it more suitable for residual/dense links, for example.
    Handles tensors references filtering (e.g. for LayerChoices candidates which takes different tensor references), NNI NAS mutable input choices, and various checks/casts of input and output tensor(s) (Ensure appropriate usage of sequences and single tensors, `TENSOR_OR_SEQ_OF_TENSORS_T`, with or without minibatch dim and with or without constraints on number of tensors in each input/referrenced sequences).
    `ForwardCallbackSubmodule`s are handled in a specific way by `DeepcvModule` for builtin support of output tensor reference(s) and NNI NAS Mutable InputChoice support (e.g. residual links with `deepcv.meta.nn_spec.yaml_tokens.FROM` parameter).

    .. See `deepcv.meta.submodule_creators.add_residual_dense_link_creator` for example usage of `ForwardCallbackSubmodule` in a submodule creator).
    TODO: Move this code to deepcv.meta.nn.forward_call_convention_dec to allow any (sub) modules to take sub output references and have NNI mutable input(s)
    """

    def __init__(self, forward_callback: SUBMODULE_FORWARD_CALLBACK_T):
        """ Instanciate a `ForwardCallbackSubmodule` PyTorch module which is defined from a callback called on forward passes. This module have a specific meaning/handling as submodule of a `deepcv.meta.base_module.DeepcvModule`.
        If `apply_parallel_forward` is `True`, then forward pass callback is called for each input tensor(s) (and if out tensor references are taken, callback will receive ith tensor of all references along with ith input tensor from previous submodule).
        If `refs_tensor_count_similar` or `in_tensors_count_similar_to_refs` are `None`, then they defaults to `apply_parallel_forward` bool value.

        NOTE: `referenced_submodules_out: List[TENSOR_OR_SEQ_OF_TENSORS_T]` argument isn't mandatory in forward callback signature from a submodule creator, but can be taken according to your needs (if tensor references and/or NNI NAS Mutable InputChoice support is needed for this NN submodule).
        NOTE: If `in_tensors_count_similar_to_refs` is `True` then `refs_tensor_count_similar` value won't have any effects as the first tensor count check implies the second one.
        """
        super().__init__()
        forward_callback_signature = inspect.signature(forward_callback).parameters
        self.forward_callback: SUBMODULE_FORWARD_CALLBACK_T = forward_callback
        self.takes_tensor_references = 'referenced_submodules_out' in forward_callback_signature

        # Reserved attribute: `self.mutable_input_choice` is filled at runtime by `DeepcvModule` when parsing architecture from NN YAML spec. (i.e. wont be `None` during forward passes if an NNI NAS input choice is specified in NN specs.)
        self.mutable_input_choice: nni_mutables.InputChoice = None
        # Reserved attribute:  `self.referenced_submodules` is filled at runtime by `DeepcvModule` when parsing architecture from NN YAML spec. (i.e. wont be `None` during forward passes if tensor reference(s) are specified in NN specs., e.g. using `deepcv.meta.nn_spec.yaml_tokens.FROM` or `deepcv.meta.nn_spec.yaml_tokens.FROM_NAS_INPUT_CHOICE`)
        self.referenced_submodules: List[str] = None

    def forward(self, tensors: TENSOR_OR_SEQ_OF_TENSORS_T, referenced_submodules_out: 'OrderedDict[str, TENSOR_OR_SEQ_OF_TENSORS_T]' = None) -> TENSOR_OR_SEQ_OF_TENSORS_T:
        """ Forward pass through module defined by a forward callback (functional torch operation encapsulated in a `torch.nn.Module (`ForwardCallbackSubmodule`).
        Handles tensors references filtering (e.g. for LayerChoices candidates which takes different tensor references), NNI NAS mutable input choices, and various checks/casts of input and output tensor(s) (Ensure appropriate usage of sequences and single tensors, `TENSOR_OR_SEQ_OF_TENSORS_T`, with or without minibatch dim and with or without constraints on number of tensors in each input/referrenced sequences).
        """
        # If needed, filter out tensor(s) references which are not used by underlying callback (e.g., may occur when using LayerChoice with multiple candidates which takes diffent tensor reference(s))
        refs = list()
        if referenced_submodules_out is not None and self.referenced_submodules is not None and self.takes_tensor_references:
            refs = [v if deepcv_nn.is_torch_obj(v) else [v, ] for n, v in referenced_submodules_out.items() if n in self.referenced_submodules]
            refs = refs if self.mutable_input_choice is None else self.mutable_input_choice(refs)
        elif referenced_submodules_out is not None or self.referenced_submodules is not None or self.takes_tensor_references:
            raise ValueError(f'Error: Uncoherent usage of output tensor references: (Did you provided `{yaml_tokens.FROM}` to a submodule which doesnt support tensor references?){NL}'
                             f'Got `referenced_submodules_out="{referenced_submodules_out}"` while `self.referenced_submodules="{self.referenced_submodules}"` and `self.takes_tensor_references="{self.takes_tensor_references}"`')
        del referenced_submodules_out

        # Apply forward pass callback (apply it for each input tensor(s) if `self.apply_parallel_forward` is `True`)
        return self.forward_callback(tensors)

#___________________________________________ SUBMODULE CREATORS FUNCTIONS _____________________________________________#


def submodule_creator_dec(name: str, submodule_creators: SUBMODULE_CREATORS_DICT_T = BASIC_SUBMODULE_CREATORS, allowed_subm_params_keys: Set[str] = None, required_subm_params_keys: Set[str] = None) -> Callable[[Callable], Callable]:
    """ Decorator helper function which appends a new entry to `submodule_creators` with decorated function associated to its `name`.
    NOTE: All keys specified in `required_subm_params_keys` will be considered as allowed (no need to specify those in `allowed_subm_params_keys`)
    """
    assert name not in submodule_creators, f'Error: "{name}" submodule creator entry already exists, can have duplicate submodule creator names.'
    if allowed_subm_params_keys is not None and required_subm_params_keys is not None:
        allowed_subm_params_keys = set(allowed_subm_params_keys) + set(required_subm_params_keys)

    def _decorator(creator: Callable[..., torch.nn.Module]) -> 'creator':
        nonlocal name, submodule_creators, allowed_subm_params_keys, required_subm_params_keys
        submodule_creators[name] = creator

        # If `allowed_subm_params_keys` have been specified, we provide `_check_submodule_params` implmentation to submodule creator (we dont patch it in order to avoid lossing needed informations from its signature)
        def _check_submodule_params(submodule_params: Dict[str, Any]):
            nonlocal name, allowed_subm_params_keys, required_subm_params_keys
            if allowed_subm_params_keys is not None:
                unkwown = [n for n in submodule_params.keys() if n not in set(allowed_subm_params_keys)]
                if len(unkwown) > 0:
                    raise ValueError(f'Error: "{unkwown}" parameter(s) are not allowed for "{name}" creator throught `submodule_params` argument (not in specified `allowed_submodule_params_keys` nor `required_subm_params_keys` values).{NL}'
                                     f' Allowed params are: "{allowed_subm_params_keys}"; Required params are "{required_subm_params_keys}"; plus eventual other creator parameters which can directly be passed as argument(s) (See "{name}" creator fn signature)')
            if required_subm_params_keys is not None:
                missing = [n for n in set(required_subm_params_keys) if n not in submodule_params]
                if len(missing) > 0:
                    raise ValueError(f'Error: Missing "{missing}" parameter(s) in "{name}" creator\'s `submodule_params` params dict.{NL}'
                                     f'Required params in "{name}" creator\'s `submodule_params` arg are: "{required_subm_params_keys}"; Allowed params are: "{allowed_subm_params_keys}"')
        creator._check_submodule_params = _check_submodule_params
        return creator
    return _decorator


@submodule_creator_dec(name='average_pooling')
def avg_pooling_creator(submodule_params: Dict[str, Any], input_shape: SIZE_OR_SEQ_OF_SIZE_T, channel_dim: int = 1) -> torch.nn.Module:
    if deepcv.meta.nn.is_torch_obj(input_shape):
        input_shape = [input_shape, ]
    spatial_dims = len(input_shape[0][channel_dim+1:])
    avg_pooling = deepcv.meta.nn.avg_pooling_nd(dims=spatial_dims, **submodule_params)

    if len(input_shape) <= 1:
        return avg_pooling
    else:
        # TODO: better implement parallel application of avg pooling (on multiple input tensors) / replace this monkey patching
        # Monkey-patch forward function with call convention allowing parallel apply. Works as long as each (parallel) tensor(s) have the same number of spatial dims (same `spatial_dims` value)
        avg_pooling.forward = deepcv.meta.nn.forward_call_convention_dec(apply_parallel_forward=True, ignore_sub_refs=True)(avg_pooling.forward)
        return avg_pooling


@submodule_creator_dec(name='reduce', allowed_subm_params_keys={})
def reduction_subm_creator(submodule_params: Dict[str, Any], fn: str, keep_dim: bool = False) -> ForwardCallbackSubmodule:
    """ Simple standalone submodule creator for reduction function usage alone.
    Applying reduction function is already possible with some other submodules with reduction function support but `reduce` submodule is standalone/lightweight and more explicit when only a reduction function on input tensors from previous submodule is needed.
    .. See `deepcv.meta.submodule_creators.TENSOR_REDUCTION_FNS` for more details.
    """
    select_tensor_creator._check_submodule_params(submodule_params)  # implemented by `submodule_creator_dec`
    return ForwardCallbackSubmodule(partial(fn, reduction=fn, keep_dim=keep_dim))


@submodule_creator_dec(name='select_tensor', required_subm_params_keys={'slice', })
def select_tensor_creator(submodule_params: Dict[str, Any], reduction: str = 'none') -> ForwardCallbackSubmodule:
    select_tensor_creator._check_submodule_params(submodule_params)  # implemetned by `submodule_creator_dec`
    parsed_slice = parse_slice(submodule_params['slice'])
    reduction = BASIC_SUBMODULE_CREATORS[reduction]

    @deepcv.meta.nn.forward_call_convention_dec(apply_parallel_forward=False)
    def _select_tensor_forward(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        nonlocal parsed_slice, reduction
        return reduction(tensors[parsed_slice])

    return ForwardCallbackSubmodule(_select_tensor_forward)


@submodule_creator_dec(name=yaml_tokens.NEW_BRANCH_FROM_TENSOR, allowed_subm_params_keys={yaml_tokens.FROM, yaml_tokens.FROM_NAS_INPUT_CHOICE})
def new_branch_creator(submodule_params: Dict[str, Any], reduction: str = 'concat') -> ForwardCallbackSubmodule:
    new_branch_creator._check_submodule_params(submodule_params)  # implemetned by `submodule_creator_dec`
    if yaml_tokens.FROM not in submodule_params or yaml_tokens.FROM_NAS_INPUT_CHOICE not in submodule_params:
        raise ValueError(f'Error: "{yaml_tokens.NEW_BRANCH_FROM_TENSOR}" submodules at least needs "{yaml_tokens.FROM}" or "{yaml_tokens.FROM_NAS_INPUT_CHOICE}" param in `submodule_params`')
    reduction = BASIC_SUBMODULE_CREATORS[reduction]

    @deepcv.meta.nn.forward_call_convention_dec(apply_parallel_forward=True, ignore_prev_subm_intput=True, refs_tensor_count_similar=True)
    def _new_branch_forward(_prev_subm_out: List[torch.Tensor], referenced_submodules_out: List[List[torch.Tensor]]) -> TENSOR_OR_SEQ_OF_TENSORS_T:
        """ Simple forward pass callback which takes referenced output tensor(s) and ignores previous submodule output features, allowing to define siamese/parallel branches thereafter.
        In other words, `deepcv.meta.nn_spec.yaml_tokens.NEW_BRANCH_FROM_TENSOR` submodules are similar to dense links but will only use referenced submodule(s) output, allowing new siamese/parrallel NN branches to be defined (wont reuse previous submodule output features)
        If multiple tensors are referenced using `deepcv.meta.nn_spec.yaml_tokens.FROM` (or `deepcv.meta.nn_spec.yaml_tokens.FROM_NAS_INPUT_CHOICE`), `reduction` reduction function will be applied.
        Reduction function is 'concat' by default and can be overriden by `reduction` parameter in link submodule spec., see `deepcv.meta.submodule_creators.TENSOR_REDUCTION_FNS` for all supported/available reduction functions.

        NOTE: All referenced tensors should have the same number of parallel tensors but, unlike residual/dense links, there are no constraints on input tensor(s) count from previous submodule
        NOTE: Can be applied in parallel, which means if tensor reference(s) each contains multiple tensors, a new branch will be created for each of those (reduction is then performed on all 'i'th tensors of each refs, for all 'i' from 0 to tensor count per ref).
        """
        nonlocal reduction
        # Ignores `_prev_subm_out` input from previous submodule output
        return reduction(referenced_submodules_out)
        # TODO: remove this once code is debugged: return [reduction(ith_refs_tensors) for ith_refs_tensors in zip(*referenced_submodules_out)]
    return ForwardCallbackSubmodule(_new_branch_forward, )


def add_nn_layer_creator(layer_op_t: Type[torch.nn.Module], creator_name: str, submodule_creators: Dict[str, Callable] = BASIC_SUBMODULE_CREATORS) -> Callable[['submodule_params', 'prev_shapes', 'act_fn', 'dropout_prob', 'batch_norm', 'channel_dim', 'preactivation'], torch.nn.Module]:
    """ Creates a fully connected or convolutional NN layer with optional dropout and batch/layer/instance/group norm support (and preactivation, activation function, ... choices)  

    NOTE: We assume here that features/inputs are given in (mini)batches (`channel_dim` defaults to 1)  
    NOTE: For convolution ops, if 'padding' isn't explicitly provided, padding is infered from `kernel_size` in order to keep feature maps shape unchanged, if possible (i.e. if `kernel_size` shape isn't uneven). See `deepcv.meta.nn.get_padding_from_kernel` for more details.  
    """
    if deepcv_nn.is_conv(layer_op_t) and not isinstance(layer_op_t, torch.nn.Linear):
        raise TypeError(f'Error: Wrong `layer_op_t` type, cant create a NN layer of type {layer_op_t} with `deepcv.meta.submodule_creators.add_nn_layer_creator` '
                        'submodule creator (`layer_op_t` should either be a convolution or a `torch.nn.Linear`).')

    @submodule_creator_dec(name=creator_name, submodule_creators=submodule_creators)
    def _nn_layer_creator(submodule_params: Dict[str, Any], input_shape: torch.Size, act_fn: torch.nn.Module = None, dropout_prob: float = None, channel_dim: int = 1, preactivation: bool = False,
                          batch_norm=None, layer_norm=None, instance_norm=None, group_norm=None, layer_nrm_and_mean_batch_nrm=None) -> torch.nn.Module:
        nonlocal layer_op_t

        # Only supports convolutions and linear layers in this submodule creator
        if 'in_features' not in submodule_params and isinstance(layer_op_t, torch.nn.Linear):
            submodule_params['in_features'] = np.prod(input_shape[channel_dim:])
        else:
            if 'padding' not in submodule_params:
                submodule_params['padding'] = deepcv_nn.get_padding_from_kernel(submodule_params['kernel_size'], warn_on_uneven_kernel=False)
            if 'in_channels' not in submodule_params:
                submodule_params['in_channels'] = input_shape[channel_dim]

        sequential_subm = deepcv_nn.layer(layer_op=layer_op_t(**submodule_params), act_fn=act_fn, dropout_prob=dropout_prob, preactivation=preactivation,
                                          input_shape=input_shape[channel_dim:], batch_norm=batch_norm, layer_norm=layer_norm, instance_norm=instance_norm, group_norm=group_norm, layer_nrm_and_mean_batch_nrm=layer_nrm_and_mean_batch_nrm)
        # Allow to apply the same convolution op on each input tensor(s) in parallel if there are multiple input tensors
        sequential_subm.forward = deepcv.meta.nn.forward_call_convention_dec(apply_parallel_forward=True, ignore_sub_refs=True)(sequential_subm.forward)
        return sequential_subm

    _nn_layer_creator.__doc__ = add_nn_layer_creator.__doc__
    return _nn_layer_creator


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

    Returns a callback throught `ForwardCallbackSubmodule` which may be used in a `DeepcvModule` NN architecture spec and called at forward pass(es).  

    Like any other DeepcvModule submodule creators which returns a forward callback with `referenced_submodules_out` arg, tensor references can be specified through `deepcv.meta.nn_spec.yaml_tokens.FROM` or `deepcv.meta.nn_spec.yaml_tokens.FROM_NAS_INPUT_CHOICE` params in YAML submodule spec.  
    A reduction function can be specified in submodule spec (See `deepcv.meta.submodule_creators.TENSOR_REDUCTION_FNS`): by default, reduction function will be 'sum' if this is a residual link and 'concat' if this is a dense link.  

    If 'allow_scaling' is `True` (when `allow_scaling: True` is specified in YAML residual/dense link spec.), then if residual/dense features have different shapes on dimensions following `channel_dim`, it will be scaled (upscaled or downscaled) using [`torch.functional.interpolate`](https://pytorch.org/docs/stable/nn.functional.html#interpolate).  
    By default interpolation mode if choosen according to spatial dim count (1D: 'linear', 2D: 'bilinear', 3D: 'trilinear', 'nearset' otherwise); If needed, you can override defaults by providing `scaling_mode` and/or `scaling_align_corners` arguments; For more details, see `deepcv.meta.nn.interpolate` and [`torch.nn.functional.interpolate` doc](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate)).  

    If `apply_in_parallel` is `True`, then, when input from previous submodule contains multiple tensors (sequence), a different residual/dense link is created for each of them ('i'th residual/dense link sums/concats 'i'th tensor of ref(s) with 'i'th tensor from previous submodule). NOTE: Thus, all referenced tensors should have the same number of parallel tensors;  
    .. See `refs_tensor_count_similar` and `in_tensors_count_similar_to_refs` args of `deepcv.meta.submodule_creators.ForwardCallbackSubmodule` for more details on those constraints and output tensor references mechanism.  

    NOTE: Scaling/interpolation of residual/dense tensors is only supported for 1D, 2D and 3D features, without taking into account channel and minibatch dimensions. Also note that `minibatch` dimension is required by [`torch.functional.interpolate`](https://pytorch.org/docs/stable/nn.functional.html#interpolate).  
    NOTE: If `allow_scaling` is `False`, output features shapes of these two or more submodules must be the same, except for the channels/filters dimension if this is a dense link.  
    NOTE: The only diference between residual and dense links ('is_residual' beeing 'True' of 'False') is the default `reduction` function beeing respectively 'sum' and 'concat' (when `None`).  
    """
    @submodule_creator_dec(name=creator_name, submodule_creators=submodule_creators, allowed_subm_params_keys={yaml_tokens.FROM, yaml_tokens.FROM_NAS_INPUT_CHOICE})
    def _link_creator(submodule_params: Dict[str, Any], allow_scaling: bool = False, scaling_align_corners: bool = False, scaling_mode: str = None, reduction: str = 'sum' if is_residual else 'concat', apply_in_parallel: bool = True, channel_dim: int = 1) -> ForwardCallbackSubmodule:
        nonlocal creator_name
        _link_creator._check_submodule_params(submodule_params)  # implemetned by `submodule_creator_dec`
        if yaml_tokens.FROM not in submodule_params and yaml_tokens.FROM_NAS_INPUT_CHOICE not in submodule_params:
            raise ValueError(f'Error: Missing "{yaml_tokens.FROM}" or "{yaml_tokens.FROM_NAS_INPUT_CHOICE}" parameter in '
                             f'{creator_name} link YAML specification; You should at least provide a tensor reference.')
        reduction = TENSOR_REDUCTION_FNS[reduction]

        @deepcv.meta.nn.forward_call_convention_dec(apply_parallel_forward=apply_in_parallel, in_tensors_count_similar_to_refs=apply_in_parallel)
        def _forward_callback(x: TENSOR_OR_SEQ_OF_TENSORS_T, referenced_submodules_out: List[TENSOR_OR_SEQ_OF_TENSORS_T]) -> TENSOR_OR_SEQ_OF_TENSORS_T:
            """ Redisual or Dense link forward pass callbacks
            If target output shape is different from one of the referenced tensor shapes, an up/down-scaling (interpolation) may be performed according to `scaling_mode` and `allow_scaling` parameters.
            NOTE: Target output shape is assumed to be the same as `x` input shape if `x` is a `torch.Tensor` or the same as the first tensor shape of `x` if `x` is a Sequence of `torch.Tensor`s.
            A reduction function can be specified (see `deepcv.meta.submodule_creators.TENSOR_REDUCTION_FNS`); If this is a residual link, reduction function defaults to 'sum' and if this is a dense link, reduction function defaults to 'concat'.
            """
            nonlocal channel_dim, reduction, creator_name
            out = [x, ] if deepcv_nn.is_torch_obj(x) else list(x)

            for refs in referenced_submodules_out:
                # If `apply_in_parallel` is False, subm output refs may contain multiple tensors (reduce all of them, flattened, along with input tensor(s))
                for y in ([refs, ] if deepcv_nn.is_torch_obj(refs) else list(refs)):
                    if out[0].shape[channel_dim + 1:] != y.shape[channel_dim + 1:]:
                        if allow_scaling:
                            # Resize y features tensor to be of the same shape as x along dimensions after channel dim (scaling performed with `torch.nn.functional.interpolate`)
                            y = deepcv_nn.interpolate(y, out[0].shape[channel_dim + 1:], scaling_mode=scaling_mode, align_corners=scaling_align_corners)
                        else:
                            raise RuntimeError(f"Error: Couldn't forward throught {creator_name} link: features from link doesn't have "
                                               f"the same shape as previous module's output shape, can't concatenate or add them. (did you forgot to allow residual/dense "
                                               f"features to be scaled using `allow_scaling: true` parameter?). `residual_shape='{y.shape}' != prev_features_shape='{out[0].shape}'`")
                    out.append(y)
            # Add or concatenate previous sub-module output features with residual or dense features
            return reduction(out)
        return ForwardCallbackSubmodule(_forward_callback)

    _link_creator.__doc__ = add_residual_dense_link_creator.__doc__
    return _link_creator


# Add Residual and Dense Link submodule creator entries to `BASIC_SUBMODULE_CREATORS`
add_residual_dense_link_creator(is_residual=True, creator_name='residual_link')
add_residual_dense_link_creator(is_residual=False, creator_name='dense_link')


#___________________________________________ SUBMODULE CREATORS UNIT TESTS ____________________________________________#

if __name__ == '__main__':
    cli = import_tests().test_module_cli(__file__)
    cli()
