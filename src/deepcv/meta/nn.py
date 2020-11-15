#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Neural Network meta module - nn.py - `DeepCV`__
Defines various neural network building blocks (layers, architectures parts, transforms, loss terms, ...)
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
    - TODO: Add EvoNorm_B0 and EvoNorm_S0 layer implentations (from deepmind neural architecture search results for normalized-activation conv layers)
"""
import copy
import enum
import math
import inspect
import logging
import functools
from argparse import SimpleNamespace
from enum import Enum
from collections import OrderedDict
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence, List, Hashable, Mapping, Generator

import numpy as np

import torch
import torch.nn
import torch.distributions as tdist  # TODO: use those distribution for helper/tooling function making easier to infer prob distribution (like Mixture Density Networks)
from hilbertcurve.hilbertcurve import HilbertCurve

from deepcv.utils import NL, NUMBER_T, get_str_repr, import_tests
from .types_aliases import *

__all__ = ['XAVIER_INIT_SUPPORTED_ACT_FN', 'ConvWithMetaLayer', 'MultiHeadConcat', 'ConcatHilbertCoords', 'ConcatCoords',
           'forward_call_convention_dec', 'func_to_module', 'to_multiscale_inputs_model', 'to_multiscale_outputs_model', 'multi_head_forward', 'concat_hilbert_coords_map', 'concat_coords_maps',
           'get_padding_from_kernel', 'nd_support', 'conv_nd', 'conv_transpose_nd', 'avg_pooling_nd', 'batch_norm_nd', 'instance_norm_nd', 'layer_norm_with_mean_only_batch_norm', 'NormTechnique',
           'NORM_TECHNIQUES_MODULES', 'normalization_techniques_impl', 'normalization_techniques', 'layer', 'resnet_net_block', 'squeeze_cell', 'multiscale_exitation_cell', 'ConvWithMetaLayer', 'meta_layer',
           'get_gain_name', 'data_parallelize', 'is_data_parallelization_usefull_heuristic', 'ensure_mean_batch_loss', 'interpolate', 'get_model_capacity', 'get_out_features_shape',
           'is_torch_obj', 'is_fully_connected', 'is_conv', 'contains_conv']
__author__ = 'Paul-Emmanuel Sotir'

#______________________________________________ NN TOOLING CONSTANTS __________________________________________________#


""" Map of default supported activation functions for `deecv.meta.nn.get_gain_name` xavier initializaton helper function
Feel free to extend it for other activation functions in order to have the right xavier init gain when using `torch.nn.init.calculate_gain(deepcv.meta.nn.get_gain_name(...))` or `deepcv.meta.base_module.DeepcvModule`
NOTE: Here are some builtin actiavion function of PyTorch which doesnt have explicit support for Xavier Init gain: 'Hardtanh', 'LogSigmoid', 'PReLU', 'ReLU6', 'RReLU', 'SELU', 'CELU', 'GELU', 'Softplus', 'Softshrink', 'Softsign', 'Tanhshrink', 'Threshold', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax', ...
"""
XAVIER_INIT_SUPPORTED_ACT_FN = {torch.nn.ReLU: 'relu', torch.nn.LeakyReLU: 'leaky_relu', torch.nn.Tanh: 'tanh', torch.nn.Sigmoid: 'sigmoid', torch.nn.Identity: 'linear'}

#_______________________________________________ NN TOOLING CLASSES ___________________________________________________#


class ConvWithMetaLayer(torch.nn.Module):
    def __init__(self, preactivation: bool = False):
        super().__init__()
        raise NotImplementedError
        self.conv = torch.nn.Conv2d(16, 3, (3, 3))  # TODO: preactivation, etc...
        normalization = dict(norm_type=..., norm_kwargs=..., input_shape=...)
        self.meta = torch.nn.Sequential(*layer(layer_op=self.conv, act_fn=torch.nn.ReLU, dropout_prob=0., preactivation=preactivation, **normalization))
        self.RANDOM_PROJ = torch.randn_like(self.conv.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        meta_out = self.meta(x)
        weight_scale, meta_out = meta_out.split((1, meta_out.size(-1) - 1), dim=-1)
        scaled_w = torch.mul(self.conv.weights.data, weight_scale)
        meta_out = meta_out.reshape((1,) * (self.conv.weights.dim() - 1) + (-1,))
        meta_out = meta_out.prod(self.RANDOM_PROJ)
        self.conv.weights = torch.add(meta_out, scaled_w)
        return self.conv(x)


# class generic_mulltiscale_class_loss(torch.nn.loss._Loss):
#     def __init__(self, reduction: str = 'mean') -> None:
#         super().__init__()
#         self._terms = [torch.nn.loss.MultiLabelMarginLoss, torch.nn.loss.BCELoss, torch.nn.loss.MultiLabelSoftMarginLoss, torch.nn.loss.KLDivLoss]

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError


# class generic_mulltiscale_class_reg_loss(torch.nn.loss._Loss):
#     TERMS = [torch.nn.loss.PoissonNLLLoss, torch.nn.loss.SmoothL1Loss, torch.nn.loss.MSELoss, torch.nn.loss.HingeEmbeddingLoss, torch.nn.loss.CosineEmbeddingLoss]

#     def __init__(self, reduction: str = 'mean', weights: torch.Tensor = torch.Tensor([1.] * len(TERMS))) -> None:
#         super().__init__()
#         self._norm_factors = torch.Tensor([1. / len(TERMS))] * len(TERMS))
#         self._weights=weights
#         self._terms=[T(reduction= reduction) for T in TERMS]

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return
#         raise NotImplementedError


# class HybridConnectivityGatedNet(deepcv.meta.base_module.DeepcvModule):
#     """ Implementation of Hybrid Connectivity Gated Net (HCGN), residual/dense conv block architecture from the following paper: https://arxiv.org/pdf/1908.09699.pdf """
#     HP_DEFAULTS = {'architecture': ..., 'act_fn': torch.nn.ReLU, 'batch_norm': None, 'dropout_prob': 0.}

#     def __init__(self, input_shape: torch.Size, hp: Dict[str, Any]):
#         """ HybridConnectivityGatedNet __init__ function
#         Args:
#             hp: Hyperparameters
#         """
#         super().__init__(input_shape, hp)
#         submodule_creators = deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS.update({'smg_module': self._smg_module_creator})
#         self.define_nn_architecture(hp['architecture'], submodule_creators)
#         self.initialize_parameters(hp['act_fn'])

#         # smg_modules = []
#         # for i, module_opts in enumerate(hp['modules']):
#         #     prev_module = smg_modules[-1]
#         #     gating = 'TODO'  # TODO: !!
#         #     raise NotImplementedError
#         #     ops = [('cell1', squeeze_cell(hp)), ('cell2', multiscale_exitation_cell(hp)), ('gating', gating)]
#         #     smg_modules.append((f'smg_module_{i}', torch.nn.Sequential(OrderedDict(ops))))
#         # self.net = torch.nn.Sequential(OrderedDict(smg_modules))

#     def forward(self, x: torch.Tensor):
#         """ Forward propagation of given input tensor through conv hybrid gated neural network
#         Args:
#             - input: Input tensor fed to convolutional neural network (must be of shape (N, C, W, H))
#         """
#         return self._net(x)

#     @staticmethod
#     def _smg_module_creator():
#         raise NotImplementedError

#_______________________________________________ NN TOOLING FUNCTIONS _________________________________________________#


def forward_call_convention_dec(apply_parallel_forward: bool = False, refs_tensor_count_similar: bool = None, in_tensors_count_similar_to_refs: bool = None, ignore_prev_subm_intput: bool = False, ignore_sub_refs: bool = False):
    """ Decorates module's forward function or functional operators in order to:  
        - , if `apply_parallel_forward` is `True`, add parallel inference over each input tensors (when input is a sequence of tensors, `forward_fn` is applied for each 'i'th input tensor along with eventual(s) `i`th tensor of each subm output reference(s) in `referenced_submodules_out`.
        - garantee that input are allays a list of tensors (or alwways a single tensor if `apply_parallel_forward` is `True`), no matter if provided `inputs` is a single tensor or a sequence (provides a `List[torch.Tensor]` (or `torch.Tensor` if `apply_parallel_forward==True`) to `forward_fn` from `TENSOR_OR_SEQ_OF_TENSORS_T` input)
        - handle eventual submodule output reference(s) and provide them to `forward_fn` throught `referenced_submodules_out` argument as a `List[List[torch.Tensor]]` if `apply_parallel_forward` is `False` or a `List[torch.Tensor]` otherwise (one tensor per reference if applied in parallel).
        - perform needed checks on submodule output reference(s), according to module/operator needs. Depending on `refs_tensor_count_similar` and `in_tensors_count_similar_to_refs` values (which defaults to `apply_parallel_forward` value if `None`), this decorator can ensure that all subm output refs each have the same number of tensors and/or the same number of tensors as `inputs` tensor(s) from previous submodule.
        - convert output from `forward_fn` by flattening it (if multiple tensors are retured for each input tensor(s)) and by returning a single tensor if there is only one output tensor (`forward_fn` have no contrainsts on whether returning a single tensor, a Sequence of tensors or a Generator of tensors).
        - can ignore input tensor(s) from previous submodule and/or submodule output tensor(s) reference(s) by providing `ignore_prev_subm_intput` or `ignore_sub_refs` bool arguments. NOTE: If `ignore_sub_refs` is `True`, then `in_tensors_count_similar_to_refs` and `refs_tensor_count_similar` have no effect. If `ignore_prev_subm_intput` is `True`, `in_tensors_count_similar_to_refs` have no effect neither.
    This decorator thus provides a simple convention over forward passes with features common to many (sub)modules/operators, allowing easier/simpler implemtenation of modules/operators without loosing genericity/expressivity (easier by providing some value checks and by allways receiving and returning tensor sequences)
    """
    refs_tensor_count_similar = apply_parallel_forward if refs_tensor_count_similar is None else refs_tensor_count_similar
    refs_tensor_count_similar = not ignore_sub_refs and refs_tensor_count_similar
    in_tensors_count_similar_to_refs = apply_parallel_forward if in_tensors_count_similar_to_refs is None else in_tensors_count_similar_to_refs
    in_tensors_count_similar_to_refs = not ignore_sub_refs and not ignore_prev_subm_intput and in_tensors_count_similar_to_refs
    # TODO: handle generators here (and verify if generators are supported according to forward_fn annotations (cast to List otherwise))
    # TODO: move parallel_apply support here

    def _decorator(forward_fn: Callable[[TENSOR_OR_SEQ_OF_TENSORS_T, ...], TENSOR_OR_SEQ_OF_TENSORS_T]) -> Callable[[TENSOR_OR_SEQ_OF_TENSORS_T, ...], TENSOR_OR_SEQ_OF_TENSORS_T]:
        nonlocal apply_parallel_forward, refs_tensor_count_similar, in_tensors_count_similar_to_refs, ignore_prev_subm_intput, ignore_sub_refs

        @functools.wraps(forward_fn)
        def _forward_wraper(inputs: TENSOR_OR_SEQ_OF_TENSORS_T, *args: Sequence[Any], referenced_submodules_out: List[TENSOR_OR_SEQ_OF_TENSORS_T] = None, **kwargs: Mapping[str, Any]) -> TENSOR_OR_SEQ_OF_TENSORS_T:
            nonlocal forward_fn, apply_parallel_forward, refs_tensor_count_similar, in_tensors_count_similar_to_refs, ignore_prev_subm_intput, ignore_sub_refs
            # NOTE: We use `is_torch_obj` instead of `isinstance(tensors, torch.Tensor)  for better perfs. Thus, we assume it is a single tensor if it is a PyTorch object (a Sequence otherwise)
            if ignore_prev_subm_intput or inputs is None:
                inputs = list()
            else:
                inputs = [inputs, ] if is_torch_obj(inputs) else list(inputs)

            if ignore_sub_refs:
                referenced_submodules_out = list()
            elif referenced_submodules_out:
                # Ensure referenced tensor(s) are stored in Sequence(s) (not single tensors(s))
                referenced_submodules_out = [[tensors, ] if is_torch_obj(tensors) else list(tensors) for tensors in referenced_submodules_out]

                # Check all referenced output tensor(s) and input tensor(s) from previous submodule each have as many tensor(s), if asked so (`in_tensors_count_similar_to_refs` and/or `refs_tensor_count_similar`)
                if in_tensors_count_similar_to_refs and not all([len(r) == len(inputs) for r in referenced_submodules_out]):
                    raise ValueError(f'Error: When `in_tensors_count_similar_to_refs` is `True`, all referenced output tensor(s) should each have as many tensor(s) as input tensor(s) from previous submodule: `len(referenced_submodules_out[i] == len(inputs) =={len(inputs)}` {NL}'
                                     f'Got `inputs={inputs}` and `referenced_submodules_out={referenced_submodules_out}`')
                elif (ignore_prev_subm_intput or not in_tensors_count_similar_to_refs) and refs_tensor_count_similar and not all([len(r) == len(referenced_submodules_out[0]) for r in referenced_submodules_out]):
                    # We check this condition only if `in_tensors_count_similar_to_refs` is `False` as checks above implies that out tensor refs all have as many tensors (unesccary to check this if the other constraint have already been checked)
                    raise ValueError(f'Error: When `refs_tensor_count_similar` is `True`, all referenced output tensor(s) should each have as many tensor(s) as the first tensor(s) ref: `len(referenced_submodules_out[0])={len(referenced_submodules_out[0])}` (or should all have only a single tensor if `refs[0]` isnt a sequence) {NL}'
                                     f'Got `inputs={inputs}` and `referenced_submodules_out={referenced_submodules_out}`')

            if not apply_parallel_forward:
                refs_kwarg = {'referenced_submodules_out': referenced_submodules_out} if referenced_submodules_out else dict()
                outs = forward_fn(inputs, *args, **refs_kwarg, **kwargs)
                outs = [outs, ] if is_torch_obj(outs) else list(outs)
            else:
                outs = list()
                # Call at least `forward_fn` once and make sure to provide sub output refs even if prev subm input is ignored
                loops_count = 1
                if not ignore_prev_subm_intput and inputs:
                    loops_count = len(inputs)
                if not ignore_sub_refs and referenced_submodules_out:
                    loops_count = len(referenced_submodules_out)
                for _i in range(loops_count):
                    # Apply forward callback by making sure to dereference tensor ref(s) and input tensor(s) which are no longer needed: Consumes `inputs` tensor(s) (and eventually `referenced_submodules_out` refs) if is a sequence using `List.pop`. (may save memory if no other refs of those tensors exists elsewhere)
                    # If there are tensor references, pass 'i'th refs tensor(s) along with 'i'th input.
                    refs_kwarg = {'referenced_submodules_out': [r.pop() for r in referenced_submodules_out]} if referenced_submodules_out else dict()
                    rslt = forward_fn(inputs.pop() if inputs else list(), *args, **refs_kwarg, **kwargs)
                    outs.extend([rslt, ] if is_torch_obj(rslt) else rslt)
            return outs if len(outs) != 1 else outs[0]
        return _forward_wraper
    return _decorator


def func_to_module(typename: str, init_params: Optional[Sequence[Union[str, inspect.Parameter]]] = None) -> Callable[[Callable], Type[torch.nn.Module]]:
    """ Returns a decorator which creates a new ``torch.nn.Module``-based class using ``forward_func`` as underlying forward function.
    Note: If ``init_params`` isn't empty, then returned ``torch.nn.Module``-based class won't have the same signature as ``forward_func``.
    This is because some arguments provided to ``forward_func`` will instead be attributes of created module, taken by class's ``__init__`` function.
    Args:
        - typename: Returned torch.nn.Module class's ``__name__``
        - init_params: An iterable of string parameter name(s) of ``forward_func`` which should be taken by class's ``__init__`` function instead of ``forward`` function.
    TODO: Test this tooling function with unit tests extensively
    """
    if init_params is None:
        init_params = []

    def _wraper(forward_func: Callable, typename: str, init_params: Sequence[Union[str, inspect.Parameter]]) -> Type[torch.nn.Module]:
        """ Returned decorator converting a function to a torch.nn.Module class
        Args:
            - forward_func: Function from which torch.nn.Module-based class is built. ``forward_func`` will be called on built module's ``forward`` function call.
        """
        if forward_func.__kwdefaults__ is not None and forward_func.__kwdefaults__ != {}:
            raise ValueError('Error: `forward_func` argument of `deepcv.meta.nn.func_to_module._wraper` function cannot have keyword defaults (`__kwdefaults__` not supported)')
        signature = inspect.signature(forward_func)
        init_params = [n if type(n) is inspect.Parameter else signature.parameters[n] for n in init_params]
        forward_params = [p for n, p in signature.parameters.items() if p not in init_params]
        init_signature = signature.replace(parameters=init_params, return_annotation=torch.nn.Module)
        forward_signature = signature.replace(parameters=forward_params, return_annotation=signature.return_annotation)

        class _Module(torch.nn.Module):
            """ torch.nn.Module generated at runtime from given forward function by `deepcv.meta.nn.func_to_module` tooling function. """
            __module__ = globals().get('__module__')

            def __init__(self, *args, **kwargs):
                super().__init__()
                bound_args = init_signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
                self._forward_args_from_init = bound_args.arguments

            @functools.wraps(forward_func)
            def forward(self, *inputs, **kwargs) -> TENSOR_OR_SEQ_OF_TENSORS_T:
                bound_args = forward_signature.bind(*inputs, **kwargs)
                bound_args.apply_defaults()
                return forward_func(**bound_args.arguments, **self._forward_args_from_init)

        prev_qualname = _Module.__qualname__ if _Module.__qualname__ not in ('', None) else _Module.__name__
        _Module.__qualname__ = (forward_func.__qualname__.replace(forward_func.__name__, typename)) if forward_func.__name__ not in ('', None) else typename
        _Module.__name__ = typename
        _Module.__doc__ = f'Module created at runtime from `{forward_func.__name__}` forward function.{NL}Initial forward function documentation:{NL}' + forward_func.__doc__
        _Module.__module__ = forward_func.__module__

        # Modify `_Module.__init__` function to match excpected signature
        init_signature = init_signature.replace(return_annotation=_Module)
        # TODO: make sure defaults are ordered the right way (should be the case as forward_signature.parameters is an OrderedDict) and make sure it plays well with `signature.apply_defaults` function
        _Module.__init__.__defaults__ = tuple(p.default for (n, p) in init_signature.parameters.items() if p.default is not None)
        _Module.__init__.__signature__ = init_signature
        _Module.__init__.__annotations__ = {n: p.annotation for n, p in init_signature.parameters.items()}
        _Module.__init__.__doc__ = f"Instanciate a new `{typename}` PyTorch module. (Class generated at runtime from `{forward_func.__name__}` forward function with `deepcv.meta.nn.func_to_module`)."
        _Module.__init__.__qualname__ = _Module.__init__.__qualname__.replace(prev_qualname, _Module.__qualname__)
        _Module.__init__.__module__ = forward_func.__module__

        # Modify `_Module.forward` function to match excpected signature
        # TODO: make sure defaults are ordered the right way (should be the case as forward_signature.parameters is an OrderedDict) and make sure it plays well with `signature.apply_defaults` function
        _Module.forward.__module__ = forward_func.__module__
        _Module.forward.__defaults__ = tuple(p.default for (n, p) in forward_signature.parameters.items() if p.default)
        _Module.forward.__signature__ = forward_signature
        _Module.forward.__qualname__ = _Module.forward.__qualname__.replace(prev_qualname, _Module.__qualname__)
        if getattr(forward_signature, '__annotations__', None) is None:
            # TODO: make sure this is usefful: `__annotations__` is already copied by `functools.wraps` but in case __annotations__ isn't defined, there may still be annotations accessible from `inspect.signature`:
            _Module.forward.__annotations__ = {n: getattr(p, 'annotation', None) for (n, p) in forward_signature.parameters.items()}
        return _Module
    return functools.partial(_wraper, typename=typename, init_params=init_params)


def to_multiscale_inputs_model(model: 'deepcv.meta.base_module.DeepcvModule', scales: int = 3, no_downscale_dims: Tuple[int] = tuple()):
    """ Turns a given deepcv module to a similar models which takes `scales` inputs at different layer depth instead of one input at first layer.
    Each new inputs are downscaled by a 2 factor, thus if you input `model` takes a 3x100x100 image the returned model will take 3 images of these respective shapes: (3x100x100; 3x50x50, 3x25x25) (assuming we have `no_downscale_dims=(0,)` and `scales=3`)
    Args:
        - model: Orginal DeepcvModule model which is modified in order to take multiple scales of input. This model should only take inputs at its first layer/sub-module
        - scales: Number of different input image scales of the returned model
        - no_downscale_dims: Input shape's dimension(s) which shouldn't be downscaled
    # TODO: add inputs transforms to downscale input 'scales' times
    # TODO: fix missing implentation parts ;->) bad but quick job here, be 🕵️‍♀️carefull🕵️‍♀️
    """
    input_shape = model._input_shape
    architecture = model._hp['architecture']
    new_hp = copy.deepcopy(model._hp)
    assert scales >= 2, f'Error: Can\'t define a multi scale input model with only ``{scales}`` (``scales`` must be greater than or equal to 2)'
    assert scales < len(architecture), f'Error: Given model ``{model}`` isn\'t deep enought for ``{scales}`` different input scales.'

    for i in range(1, scales):
        raise NotImplementedError
        submodule_indice = i * (len(architecture) // scales)
        ith_scale_submodule = architecture[i]
        append_input = {'append': '&input_{i}'}
        new_hp['architecture'].insert(submodule_indice + i - 1, append_input)

    return type(model)(model._input_shape, new_hp)


def to_multiscale_outputs_model(model: 'deepcv.meta.base_module.DeepcvModule', scales: int = 3, no_downscale_dims: Tuple[int] = tuple()):
    """
    TODO: similar implementation than to_multiscale_inputs_model
    """
    raise NotImplementedError


def multi_head_forward(tensor: TENSOR_OR_SEQ_OF_TENSORS_T, heads: Iterable[torch.nn.Module], concat_dim: int = 1, new_dim: bool = True, one_tensor_per_head: bool = False) -> torch.Tensor:
    """ Forwards `tensor` tensor throught multiple head modules: contenates each given head module's output over features first dimension or a new dimension
    Args:
        - tensor: input tensor(s) to be forwarded through head modules
        - heads: Head module taking `tensor` tensor as input and which output is concatenated over other heads dimension. All head modules must have the same output shape in order to be concatenated into output tensor (except on first features/`embedding_shape` dimension if `new_dim` is `False`)
        - concat_dim: By default, equals to `1`, which means that output tensor will be a concanetation of head's outputs tensors over 2nd dimension (typically, after batch dimension)
        - new_dim: Whether create a new concatenation dim or not. (defaults to `True`). For example, if `tensor` tensor is a batch of images or convolution outputs with `concat_dim=1` (channel dim), then if `new_dim=False` head modules output is stacked over channel dimension, otherwise output tensors are concatenated over a new dimension.
    """
    if one_tensor_per_head and not isinstance(tensor, torch.Tensor):
        if not len(tensor) >= len(heads):
            raise ValueError(f'Error in {get_str_repr(multi_head_forward)} As given `tensor`s is a sequence of tensors, and `one_tensor_per_head=True`, '
                             f'`len(tensor)={len(tensor)}` should be greater or equal to `len(heads)={len(heads)}`, got')
        outs = [head(x) for head, x in zip(heads, tensor)]
    else:
        outs = [head(tensor) for head in heads]

    return torch.cat(outs, dim=concat_dim) if new_dim else torch.stack(outs, dim=concat_dim)


def concat_coords_maps(x: TENSOR_OR_SEQ_OF_TENSORS_T, channel_dim: int = 1) -> TENSOR_OR_SEQ_OF_TENSORS_T:
    """ Concats N new features maps of euclidian coordinates (1D, 2D, ..., ND coordinates if `x` has N dimensions after `channel_dim`'s dimension) into given `x` tensor. Coordinates are concatenated at `channel_dim` dimention of `x` tenseor
    Args:
        - x: Input tensor(s) which have at least 1 dimension after `channel_dim`'s dimension
        - channel_dim: Channel dimension index in `x` tensor(s) at which coordinates maps will be concatenated (0 by default). Supports negative dim index: `channel_dim` must be in `]x.dim(); -1[ U ]-1 ; x.dim()[` range.
    Returns tensor(s) which is/are the concatenation of `x` tensor(s) with coordinates feature map(s) at `channel_dim` dimension, allowing, for ex., to append pixel location information explicitly into data proceesed in your model(s).
    .. See also `deepcv.meta.nn.concat_hilbert_coords_map` (or its respective module `deepcv.meta.nn.ConcatHilbertCoords`) which is simmilar to `concat_coords_maps` (or `ConcatCoords` module alternative) but will only concatenate one coordinates map of Hilbert Curve coordinates/distance, no matter how many dimensions `x` have (location information takes less memory space by using Hilbert space filling curve instead of euclidian coordinates).

    Bellow is an axample of generated euclidian coordinates maps in case of 2D features:
    ```
    # If `x` is a tensor of shape (N, C, H, W) and `channel_dim` is 1, then euclidian coordinates maps which will be concatenated to `x` will be:
        0, 0    1, 0    ...     W , 0

        0, 1    1, 1    ...     W, 1

        ...     ...     ...     ...

        0, H    1, H    ...     H, W
    ```

    """
    return _concat_coords_maps_impl(x, channel_dim=channel_dim, euclidian=True)


def concat_hilbert_coords_map(x: TENSOR_OR_SEQ_OF_TENSORS_T, channel_dim: int = 1) -> TENSOR_OR_SEQ_OF_TENSORS_T:
    """ Concatenates to feature maps of `x` tensor(s) a new channel which contains position information using Hilbert curve distance metric.
    This operation is close to `concat_coords_maps`, except that we only append one channel of hilbert distance instead of N channels of euclidian coordinates (e.g. 2 channel for features from a 2D convolution).
    Args:
        - x: Input tensor(s) of N-D Feature maps with channel dimmension located at `channel_dim`th dim and feature maps dims located after channel's one. (Hilbert curve distance can be computed for any number, N, of feature map dimensions)
        - channel_dim: Channel dimension index, 1 by default.
    # TODO: cache hilbert curve to avoid to reprocess it too often
    # TODO: Parallelize when multiple tensors are provided in `x`
    """
    return _concat_coords_maps_impl(x, channel_dim=channel_dim, euclidian=False)


@deepcv.meta.nn.forward_call_convention_dec(apply_parallel_forward=True, ignore_sub_refs=True)
def _concat_coords_maps_impl(x: torch.Tensor, channel_dim: int = 1, euclidian: bool = True) -> torch.Tensor:
    """ Implementation of `concat_coords_maps` and `concat_hilbert_coords_channel`
    TODO: Add support for normalization of coordinates map(s) (normalization: Optional[...] = None argument)
    TODO: Implement unit testing for this function
    """
    if channel_dim not in range(1-x.dim(), -1) and channel_dim not in range(0, x.dim()-1):
        raise ValueError(f'Error: Invalid argument: `channel_dim` must be in "]x.dim(); -1[ U ]-1 ; x.dim()[" range, got channel_dim={channel_dim}')
    if x.shape[channel_dim+1:] not in (1, 3, 3):
        raise ValueError(f'Error: {get_str_repr(_concat_coords_maps_impl, __file__)} only support 2D or 3D input features '
                         f'(e.g. `x` features dim of 4 or 5 with minibatch and channel dims), but got `x.dim()={x.dim()}`.')

    if channel_dim < 0:
        channel_dim += x.dim()

    feature_map_size = x.shape[channel_dim + 1:]

    if euclidian:
        # Concats N feature maps which contains euclidian coordinates (N being `len(feature_map_size)`, i.e. 1D, 2D or 3D coordinates)
        coords = [torch.torch.arange(start=0, end=size - 1, step=1, dtype=x.dtype, device=x.device) for size in feature_map_size]
        coords = [c.expand(feature_map_size).view([1] * (channel_dim + 1) + [*feature_map_size]) for c in coords]
        return torch.cat([x, *coords], dim=channel_dim)
    else:
        # Concats a single feature map which contains Hilbert curve coordinates
        space_filling = HilbertCurve(n=len(feature_map_size), p=np.max(feature_map_size))
        space_fill_coords_map = np.zeros(feature_map_size)
        for coords in np.ndindex(feature_map_size):
            space_fill_coords_map[coords] = space_filling.distance_from_coordinates(coords)
        space_fill_coords_map = torch.from_numpy(space_fill_coords_map).view([1] * (channel_dim + 1) + [*feature_map_size])
        return torch.cat([x, space_fill_coords_map], dim=channel_dim)


# Torch modules created from their resective forward function:
MultiHeadConcat = func_to_module('MultiHeadConcat', init_params=['heads', 'concat_dim', 'new_dim'])(multi_head_forward)
ConcatHilbertCoords = func_to_module('ConcatHilbertCoords', init_params=['channel_dim'])(concat_hilbert_coords_map)
ConcatCoords = func_to_module('ConcatCoords', init_params=['channel_dim'])(concat_coords_maps)


def get_padding_from_kernel(kernel_size: SIZE_N_T, warn_on_uneven_kernel: bool = False) -> SIZE_N_T:
    is_sequence = not isinstance(kernel_size, Sequence)
    padding = [max(0, math.floor((ks - 1.) / 2.)) for ks in kernel_size]

    if warn_on_uneven_kernel and any([v % 2 != 0 for v in (kernel_size if is_sequence else [kernel_size, ])]):
        logging.warn(f'Warning: `kernel_size={kernel_size}` has uneven size, which may result in inapropriate output tensor shape even with "{padding}" zero-padding')
    return padding if is_sequence else padding[0]


def nd_support(_nd_types: Dict[int, Union[Callable, Type]], dims: int, *args, _name: str = None, **kwargs) -> Any:
    """ Helper function allowing easier support for N-D operations/modules, see example usage bellow for better understanding (e.g. `deepcv.meta.nn.batch_norm_nd`, or `deepcv.meta.nn.conv_nd`, ...). """
    if dims not in _nd_types:
        available_ops = ', '.join([f'{dim}D: {op.__name__ if isinstance(op, Type) else str(op)}' for dim, op in _nd_types.items()])
        raise ValueError(f'Error: {"This operator/module" if _name is  None else _name} doesnt support operations on {dims}D features maps, available ops are: `nd_types="{available_ops}"`'
                         f'(No {dims}D type/callable entry in `nd_types` of `deepcv.meta.nn.nd_support{f"(_name={_name}, ...)" if _name is not None else ""}`).')
    return _nd_types[dims](*args, **kwargs)


""" N-D Convolution operator based on `torch.nn.Conv*d` for 1D, 2D and 3D support """
conv_nd = functools.partial(nd_support, nd_types={1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}, _name='conv_nd')
""" N-D Transposed Convolution operator based on `torch.nn.ConvTranspose*d` for 1D, 2D and 3D support """
conv_transpose_nd = functools.partial(nd_support, nd_types={1: torch.nn.ConvTranspose1d, 2: torch.nn.ConvTranspose2d, 3: torch.nn.ConvTranspose3d}, _name='conv_transpose_nd')
""" N-D Average Pooling """
avg_pooling_nd = functools.partial(nd_support, nd_types={1: torch.nn.AvgPool1d, 2: torch.nn.AvgPool2d, 3: torch.nn.AvgPool3d}, _name='avg_pooling_nd')
""" N-D Batch Normalization module based on `torch.nn.BatchNorm*d` for 1D, 2D and 3D support """
batch_norm_nd = functools.partial(nd_support, _nd_types={1: torch.nn.BatchNorm1d, 2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}, _name='batch_norm_nd')
""" N-D Instance Normalization module (a.k.a Constrast Normalization) based on `torch.nn.InstanceNorm*d` for 1D, 2D and 3D support """
instance_norm_nd = functools.partial(nd_support, nd_types={1: torch.nn.InstanceNorm1d, 2: torch.nn.InstanceNorm2d, 3: torch.nn.InstanceNorm3d}, _name='instance_norm_nd')


def layer_norm_with_mean_only_batch_norm(input_shape: SIZE_OR_SEQ_OF_SIZE_T, eps=1e-05, elementwise_affine: bool = True, momentum: float = 0.1, track_running_stats: bool = True) -> torch.nn.Sequential:
    """ LayerNorm used along with 'mean-only' BatchNorm, as described in [`LayerNorm` paper](https://arxiv.org/pdf/1602.07868.pdf).
    NOTE: `input_shape` tensor shape(s) should only contain spatial dimensions and channel dim, not minibatch dim (i.e. channel dim should be 0th/first dim).
    """
    layer_norm_op = torch.nn.LayerNorm(normalized_shape=input_shape[1:], eps=eps, elementwise_affine=elementwise_affine)
    # TODO: ensure this is mean-only BatchNorm!
    mean_only_batch_norm = batch_norm_nd(num_features=input_shape[0], dims=len(input_shape[1:]), eps=eps, momentum=momentum, affine=True, track_running_stats=track_running_stats)
    return torch.nn.Sequential([layer_norm_op, mean_only_batch_norm])


class NormTechnique(enum.Enum):
    BATCH_NORM = r'batch_norm'
    LAYER_NORM = r'layer_norm'
    INSTANCE_NORM = r'instance_norm'
    GROUP_NORM = r'group_norm'
    # Local Response Norm (Normalize across channels by taking into account `size` neightbouring channels; assumes channels is the 2nd dim). For more details, see https://pytorch.org/docs/master/generated/torch.nn.LocalResponseNorm.html?highlight=norm#torch.nn.LocalResponseNorm
    LOCAL_RESPONSE_NORM = r'local_response_norm'
    # `LAYER_NORM_WITH_MEAN_ONLY_BATCH_NORM` is a special case where LayerNorm is used along with 'mean-only' BatchNorm (as described in `LayerNorm` paper: https://arxiv.org/pdf/1602.07868.pdf)
    LAYER_NORM_WITH_MEAN_ONLY_BATCH_NORM = r'layer_nrm_and_mean_batch_nrm'


NORM_TECHNIQUES_MODULES = {NormTechnique.BATCH_NORM: deepcv.meta.nn.batch_norm_nd, NormTechnique.LAYER_NORM: torch.nn.LayerNorm, NormTechnique.INSTANCE_NORM: deepcv.meta.nn.instance_norm_nd,
                           NormTechnique.GROUP_NORM: torch.nn.GroupNorm, NormTechnique.LOCAL_RESPONSE_NORM: torch.nn.LocalResponseNorm, NormTechnique.LAYER_NORM_WITH_MEAN_ONLY_BATCH_NORM: layer_norm_with_mean_only_batch_norm}


def normalization_techniques_impl(norm_type: Union[NormTechnique, Sequence[NormTechnique]], norm_kwargs: Union[Sequence[Dict[str, Any]], Dict[str, Any]], input_shape: torch.Size = None, supported_norm_ops: NORM_TECHNIQUES_MODULES_T = NORM_TECHNIQUES_MODULES) -> List[torch.nn.Module]:
    """ Creates needed `torch.nn.Module` operations to perform one or more normalization technique(s) according to provided arguments, especially `norm_type` (see `NORM_TECHNIQUES_MODULES` enum) and `norm_kwargs` (Keyword arguments dict(s) given to their respective normalization Module).
    Args:
        - norm_type: Normalization technique(s) to be used, specified as string(s) or `NormTechnique` enum value(s) (must have a corresponding entry in `supported_norm_ops`)
        - norm_kwargs: Keyword arguments dict(s) to be given to respective normalization module constructor
        - input_shape: Input tensor shape on which normalization(s) are performed, without eventual minibatch dim (i.e. `input_shape` must be the shape of input tensor starting from channel dim followed by normalized features shape, e.g. set `input_shape="(C, H, W)"` if normalized input tensor has `(N, C, H, W)` shape). If this argument is provided (not `None`), then InstanceNorm/BatchNorm's `num_features`, LayerNorm's `normalized_shape` and GroupNorm's `num_channels` args are automatically specified and wont be needed in `norm_kwargs`.
        - supported_norm_ops: Supported normalization modules/ops dict; Defaults to `deepcv.meta.nn.NORM_TECHNIQUES_MODULES`.

    Supported normalization techniques:
    - `torch.nn.BatchNorm*d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)` (through `deepcv.meta.nn.batch_norm_nd`)
    - `torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True)`
    - `torch.nn.InstanceNorm*d(num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False)` (through `deepcv.meta.nn.instance_norm_nd`)
    - `torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True)`
    - `torch.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1.0)`
    - And `deepcv.meta.nn.layer_norm_with_mean_only_batch_norm`: A special case where LayerNorm is used along with 'mean-only' BatchNorm (as described in `LayerNorm` paper: https://arxiv.org/pdf/1602.07868.pdf)

    Returns a list of `torch.nn.Module`(s) for normalization operation(s) as described by `norm_type`(s) and `norm_kwargs`

    NOTE: If `input_shape` arg isn't `None`, `instance_norm_nd`/`batch_norm_nd`'s `num_features` and `dims` args, LayerNorm's `normalized_shape` arg, GroupNorm's `num_channels` arg and `layer_norm_with_mean_only_batch_norm`'s `input_shape` argument are automatically specified from given features shape and wont be needed in `norm_kwargs` (only need to give other args in `norm_kwargs` for underlying norm `torch.nn.Module` constructor)
    NOTE: You cant specify the same normalization technique multiple times and `norm_type` and `norm_kwargs` must have the same lenght if those are `Sequence`s (multiple normalization ops)
    NOTE: According to results from Switchable-Normalization (SN) 2018 paper (https://arxiv.org/pdf/1811.07727v1.pdf):
        Instance normalization are used more often in earlier layers, batch normalization is preferred in the middle and layer normalization is used in the last more often.
        Intuitively, smaller batch sizes lead to a preference towards layer normalization and instance normalization (or Group Norm which is in between).
        .. See also following blog post about DNN normalization techniques: https://medium.com/techspace-usict/normalization-techniques-in-deep-neural-networks-9121bf100d8 (Note above inspired from it)
    TODO: Eventually setup warnings in cases where norm strategies doesnt seems compatible together or redundant (e.g.: May not play well = shape of normalized features not sufficent, ..., redundant = instance norm + another more general normalization, or group norm and layer norm and cases depending on parameters like group norm with only 1 group <=> Layer norm, Goup norm with C groups <=> Instance Norm, ...)
    TODO: Implement unit testing for this function
    """
    if isinstance(norm_type, (NormTechnique, str)):
        norm_type, norm_kwargs = [norm_type, ], [norm_kwargs, ]
    if len(set(norm_type)) != len(norm_type):
        raise ValueError(f'Error: Cant use the same normalization technique mutiple times at once (duplicates forbiden in `norm_type` argument; Got `norm_type(s)="{norm_type}"`')
    if not isinstance(norm_type, Sequence) or not isinstance(norm_kwargs, Sequence) or len(norm_type) != len(norm_kwargs):
        raise TypeError('Error: `norm_type` and `norm_kwargs` must either both be a sequence of the same size or both only one normalization technique and one keyword args dict; '
                        f'Got `norm_type(s)="{norm_type}"` and `norm_kwargs="{norm_kwargs}"`')

    norm_ops = list()
    for norm_t, kwargs in zip(norm_type, norm_kwargs):
        # If `input_shape` is not `None, provide InstanceNorm/BatchNorm's `num_features`, LayerNorm's `normalized_shape` and GroupNorm's `num_channels` kwargs (or `input_shape` for `deepcv.meta.nn.layer_norm_with_mean_only_batch_norm`)
        if input_shape is not None:
            if norm_t in (NormTechnique.INSTANCE_NORM, NormTechnique.BATCH_NORM):
                kwargs['num_features'] = input_shape[0]
                if len(input_shape) > 1:
                    # If given `input_shape` isn't missing dims other than channel dim, we also provide `dims` argument of `deepcv.meta.nn.batch_norm_nd` or `deepcv.meta.nn.instance_norm_nd`
                    kwargs['dims'] = len(input_shape) - 1
            if norm_t == NormTechnique.LAYER_NORM:
                kwargs['normalized_shape'] = input_shape[1:]
            elif norm_t == NormTechnique.GROUP_NORM:
                kwargs['num_channels'] = input_shape[0]
            elif norm_t == NormTechnique.LAYER_NORM_WITH_MEAN_ONLY_BATCH_NORM:
                kwargs['input_shape'] = input_shape

        if norm_t not in supported_norm_ops:
            raise ValueError(F'Error: "{norm_t}" is an unkown or forbiden normalization technique: It isn\'t specified in `supported_norm_ops="{supported_norm_ops}"`')
        norm_ops.append(supported_norm_ops[norm_t](**kwargs))
    return norm_ops


def normalization_techniques(input_shape: torch.Size = None, batch_norm: Dict[str, Any] = None, layer_norm: Dict[str, Any] = None, instance_norm: Dict[str, Any] = None, group_norm: Dict[str, Any] = None, layer_nrm_and_mean_batch_nrm: Dict[str, Any] = None) -> List[torch.nn.Module]:
    """ Creates needed `torch.nn.Module` operations to perform one or more normalization technique(s) according to provided keyword args dict(s).
    .. See `deepcv.meta.nn.normalization_techniques_impl` for more details on this function and normalization techniques.
    Returns a list of `torch.nn.Module`(s) for normalization operation(s) as described by `norm_type`(s) and `norm_kwargs`
    """
    # Handle specified normalization techniques if any (BatchNorm, LayerNorm, InstanceNorm, GroupNorm and/or ln_with_mean_bn)
    norm_techniques = {NormTechnique.BATCH_NORM: batch_norm, NormTechnique.LAYER_NORM: layer_norm, NormTechnique.INSTANCE_NORM: instance_norm,
                       NormTechnique.GROUP_NORM: group_norm, NormTechnique.LAYER_NORM_WITH_MEAN_ONLY_BATCH_NORM: layer_nrm_and_mean_batch_nrm}
    norms = {t: args for t, args in norm_techniques.items() if args is not None and len(args) > 0}
    if len(norms) <= 0:
        return list()
    return normalization_techniques_impl(norm_type=norms.keys(), norm_kwargs=norms.values(), input_shape=input_shape, supported_norm_ops=NORM_TECHNIQUES_MODULES)


def layer(layer_op: torch.nn.Module, act_fn: Optional[Type[torch.nn.Module]], dropout_prob: float = None, preactivation: bool = False,
          input_shape: torch.Size = None, **norms_kwargs: Dict[str, Any]) -> torch.nn.Module:
    """ Defines neural network layer operations
    Args:
        - layer_op: Layer operation to be used (e.g. torch.nn.Conv2d, torch.nn.Linear, ...).
        - act_fn: Activation function (If `None`, then no activation function is used)
        - dropout_prob: Dropout probability (if dropout_prob is None or 0., then no dropout ops is used)
        - preactivation: Boolean specifying whether to use preactivatation operation order: "(?dropout) - (?BN) - Act - Layer" or default operation order: "(?Dropout) - Layer - Act - (?BN)"
        - input_shape/norms_kwargs: Only needed for normalization technique(s); See respective argument of `deepcv.meta.nn.normalization_techniques` and ``deepcv.meta.nn.normalization_techniques_impl` for more details (If all `None`, then `deepcv.meta.nn.normalization_techniques` won't be called). NOTE: `norms_kwargs` can contain the following keyword arguments: "batch_norm/layer_norm/instance_norm/group_norm/layer_nrm_and_mean_batch_nrm".

    Returns a `torch.nn.Sequential` module which contains layer operations as described by provided arguments 
    NOTE: Dropout used along with batch norm may be unrecommended (see respective warning message).
    """
    if not hasattr(layer_op, 'weight'):
        raise ValueError(f'Error: Bad layer operation module argument, no `weight` attribute found in layer_op="{layer_op}"')

    def _dropout() -> Optional[torch.nn.Module]:
        if dropout_prob is not None and dropout_prob != 0.:
            if NormTechnique.BATCH_NORM in norm_type or NormTechnique.BATCH_NORM == norm_type:
                logging.warn("""Warning: Dropout used along with normalization technique(s), like BatchNorm, may be unrecommended, see
                                [CVPR 2019 paper: 'Understanding the Disharmony Between Dropout and Batch Normalization by Variance'](https://zpascal.net/cvpr2019/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf)""")
            return torch.nn.Dropout(p=dropout_prob)
        return None

    # Handle normalization operations using `deepcv.meta.nn.normalization_techniques`
    norm_ops = list()
    if any(norms_kwargs.values()):
        if not preactivation:
            # Postactivation; Normalization is applied after convolution, so we need to find tensor shape by applying conv and act_fn ops to a mock/dummy tensor of shape `input_shape`
            dummy_in_tensor = torch.zeros(input_shape)
            input_shape = get_out_features_shape(input_shape, torch.nn.Sequential(layer_op, act_fn), use_minibatches=True)[1:]
        norm_ops = normalization_techniques(input_shape, **norms_kwargs)

    # Return a `torch.nn.Sequential` of layer operations (ops order differs depends on `preactivation` bool value)
    ops = (_dropout(), *norm_ops, act_fn(), layer_op) if preactivation else (_dropout(), layer_op, act_fn(), *norm_ops)
    return torch.nn.Sequential(*(m for m in ops if m is not None))


def resnet_net_block(hp: Union[Dict[str, Any], 'deepcv.meta.hyperparams.Hyperparameters']) -> torch.nn.Module:
    raise NotImplementedError
    ops = [('conv1', layer(**hp['conv'])), ('conv2', layer(**hp['conv']))]
    return torch.nn.Sequential(OrderedDict(ops))


def squeeze_cell(hp: HYPERPARAMS_T) -> torch.nn.Module:
    raise NotImplementedError


def multiscale_exitation_cell(hp: HYPERPARAMS_T) -> torch.nn.Module:
    raise NotImplementedError


def meta_layer(input_feature_shape: torch.Size, target_module: torch.nn.Parameter):
    """ A 'parallel'/'meta' layer applied to previous layer/block's features to infer global statistics of next layer's weight matrix
    Args:
        - layer_op: Underlying layer operation module
    """
    raise NotImplementedError
    conv = torch.nn.Conv2d(16, 3, (3, 3))
    normalization = dict(norm_type=..., norm_kwargs=..., input_shape=...)
    underlying_layer_ops = layer(layer_op=conv, act_fn=torch.nn.ReLU, dropout_prob=None, preactivation=False, **normalization)
    ops = [('underlying_layer_ops', underlying_layer_ops), ]

    return torch.nn.Sequential(OrderedDict(ops))


def get_gain_name(act_fn: Type[torch.nn.Module], default: str = 'relu', supported_act_fns: Dict[Type[torch.nn.Module], str] = XAVIER_INIT_SUPPORTED_ACT_FN) -> str:
    """ Intended to be used with torch.nn.init.calculate_gain(str):
    NOTE: For `leaky_relu` (and any other parametric act fn with future support), `torch.nn.init.calculate_gain` needs another argument `param` which should be act fn parameter (Leaky ReLU slope)

    Example usage:
    ``` python
        weights_to_init = ...
        gain = torch.nn.init.calculate_gain(get_gain_act_name(torch.nn.ReLU))
        # You can alternatively use `torch.nn.init.xavier_uniform` according to your needs (e.g. linear layer weights init instead of convolution's)
        torch.nn.init.xavier_normal(weights_to_init, gain=gain)
    ```

    .. See (`torch.nn.init.calculate_gain` PyTorch documentation)[https://pytorch.org/docs/1.5.0/nn.init.html?highlight=calculate_gain#torch.nn.init.calculate_gain] for more details.
    """
    if act_fn in supported_act_fns:
        return supported_act_fns[act_fn]
    else:
        logging.warn(f'Warning: Unsupported activation function "{act_fn}", defaulting to "{default}" xavier initialization. '
                     f'(You may need to add support for "{act_fn}" xavier init gain through `supported_act_fns` arg or its default `XAVIER_INIT_SUPPORTED_ACT_FN`).{NL}'
                     f'Supported activation function are: supported_act_fns="{supported_act_fns}"')
        return default


def data_parallelize(model: torch.nn.Module, print_msg: bool = True) -> torch.nn.Module:
    """ Make use of all available GPU using torch.nn.DataParallel if there are multiple GPUs available
    NOTE: ensure to be using different random seeds for each process if you use techniques like data-augmentation or any other techniques which needs random numbers different for each steps. TODO: make sure this isn't already done by Pytorch?
    """
    if torch.cuda.device_count() > 1:
        print(f'> Using "torch.nn.DataParallel({model})" on {torch.cuda.device_count()} GPUs.')
        model = torch.nn.DataParallel(model)
    return model


def is_data_parallelization_usefull_heuristic(model: torch.nn.Module, batch_shape: torch.Size, print_msg: bool = True) -> bool:
    """ Returns whether if data parallelization could be helpfull in terms of performances using a heuristic from model capacity, GPU count, batch_size and dataset's shape
    Args:
        - model: Model to be trained (computes its parameters capacity)
        - batch_shape: Dataset's batches shape
    TODO: perform a random/grid search to find out optimal factors (or using any other black box optimization techniques)
    # TODO: improve `gpus_score` heuristic score term according to GPU bandwidth and FLOPs?
    """
    ngpus = torch.cuda.device_count()
    capacity_factor, batch_factor, ngpus_factor = 1. / (1024 * 1024), 1. / (1024 * 512), 1. / 8.
    if ngpus <= 1:
        return False
    capacity_score = 0.5 * torch.sigmoid(torch.log10(capacity_factor * torch.FloatTensor([get_model_capacity(model) + 1.]))) / 5.
    batch_score = 3. * torch.sigmoid(torch.log10(np.log10(batch_factor * torch.FloatTensor([np.prod(batch_shape) + 1.])) + 1.)) / 5.
    gpus_score = 1.5 * torch.sigmoid(torch.log10(ngpus_factor * torch.FloatTensor([ngpus - 1.]) + 1.)) / 5.
    heuristic = float(capacity_score + batch_score + gpus_score)
    if print_msg:
        may_or_wont, lt_gt_op = ('may', '>') if heuristic > 0.5 else ('wont', '<')
        logging.info(f'DataParallelization {may_or_wont} be helpfull to improve training performances: heuristic({heuristic:.3f}) {lt_gt_op} 0.5 (heuristic({heuristic:.3f}) = capacity_score({float(capacity_score):.3f}) + batch_score({float(batch_score):.3f}) + gpus_score({float(gpus_score):.3f}))')
    return heuristic > 0.5


def ensure_mean_batch_loss(loss: LOSS_FN_T, batch_loss: Union[NUMBER_T, torch.Tensor, Sequence[NUMBER_T]], sum_divider: FLOAT_OR_FLOAT_TENSOR_T, dtype: Optional[torch.dtype] = torch.float) -> torch.FloatTensor:
    """ Ensures `batch_loss` tensor resulting from given `loss` is a mean whatever `loss.reduction` is used (allways returns the mean of loss(es) across minibatch dim)
    NOTE: Make sure `sum_divider` is the (mini)batch size in order to obtain a valid mean when `loss.reduction` is `sum`.
    TODO: Raise or warn if tensor caontains `Nan` or +/-`inf` value(s)?
    """
    if isinstance(batch_loss, (NUMBER_T, Sequence)):
        # If `batch_loss` is a Python builtin number (int, float, ...) or a Sequence, convert it back to a tensor
        batch_loss = torch.Tensor(batch_loss if isinstance(batch_loss, Sequence) else [batch_loss, ])

    # Convert `batch_loss` data type if `dtype` is not `None`
    batch_loss = batch_loss if dtype is None else batch_loss.to(dtype)

    # Make sure to return a mean loss from `batch_loss` by performing proper ops according to `loss.reduction` value (and eventually convert it to `dtype`)
    if batch_loss.shape == torch.Size([0, ]):
        return batch_loss  # Return empty tensor as is
    if loss.reduction == 'none':
        return torch.mean(batch_loss)
    elif loss.reduction == 'sum':
        return torch.div(batch_loss, sum_divider)
    elif loss.reduction == 'mean':
        return batch_loss  # Return `batch_loss` as is (already reduced into a mean)
    raise ValueError(f'Error Unsupported "{loss.reduction}" reduction value in `deepcv.meta.nn.ensure_mean_batch_loss`{NL}'
                     f'Got: `loss.reduction="{loss.reduction}"`;`batch_loss="{batch_loss}"`; `sum_divider="{sum_divider}"`, `dtype="{dtype}"`')


def interpolate(tensors: TENSOR_OR_SEQ_OF_TENSORS_T, out_spatial_shape: torch.Size, scaling_mode: str = None, align_corners: bool = False) -> TENSOR_OR_SEQ_OF_TENSORS_T:
    """ Simple helper function for easier usage of `torch.nn.functional.interpolate` with interpolation mode depending on spatial dim count and up/down-sampling multiple tensor(s) `tensors`.
    NOTE: Scaling/interpolation mode is choosen automatically but can be overriden by `scaling_mode` argument if not `None`.
    """
    if scaling_mode is None:
        scaling_mode = 'linear' if len(out_spatial_shape) == 1 else ('bilinear' if len(out_spatial_shape) == 2 else ('trilinear' if len(out_spatial_shape) == 3 else 'nearest'))

    def _interpolate(x: torch.Tensor) -> torch.Tensor:
        nonlocal scaling_mode, out_spatial_shape, align_corners
        if x.shape[-len(out_spatial_shape):] != out_spatial_shape:
            return torch.nn.functional.interpolate(x, size=out_spatial_shape, mode=scaling_mode, align_corners=align_corners)
    return return_tensors_or_single_tensor(tensors, process_fn=_interpolate)


def get_model_capacity(model: Optional[torch.nn.Module]):
    """ Returns given `model`'s capacity (total parameter count)
    TODO: add `only_learnable` bool argument to only take into account learnable parameters, if possible
    TODO: Modify `deepcv.meta.nn.get_model_capacity` in case submodule is a `MutableLayer`s (Return a list of capacities or mean capacity? For now, returns sum of capacities...)?
    """
    if model is None:
        return 0
    return sum([np.prod(param.shape) for param in model.parameters(recurse=True)])


def get_out_features_shape(input_shape: SIZE_OR_SEQ_OF_SIZE_T, module: torch.nn.Module, use_minibatches: bool = True) -> Union[torch.Size, List[torch.Size], Dict[Hashable, torch.Size]]:
    """ Performs a forward pass with a dummy input tensor to figure out module's output shape.
    NOTE: `input_shape` is assumed to be input tensor shape without eventual minibatch dim: If `use_minibatches` is `True`, input tensor will be unsueezed to have a minibatch dim, along with `input_shape` dims, before being forwarded throught given `module`.
    Returns output tensor shape(s) of given `module` applied to a dummy input (`torch.nn.zeros`) with or without additional minibatch dim (depending on `use_minibatches`). If `module` returns a `Sequence` or a `Dict` of `torch.Tensor` instead of a single tensor, this function will return a list or a dict of output tensors shapes
    """
    single_input_shape = isinstance(input_shape[0], NUMBER_T)
    if single_input_shape:
        input_shape = [input_shape, ]

    module.eval()
    with torch.no_grad():
        dummy_x = [(torch.unsqueeze(torch.zeros(s), dim=0) if use_minibatches else torch.zeros(s)) for s in input_shape]
        outputs = module(dummy_x[0] if single_input_shape else dummy_x)
        if isinstance(outputs, torch.Tensor):
            return outputs.shape
        return {n: r.shape for n, r in outputs.items()} if isinstance(outputs, Dict) else [r.shape for r in outputs]


def is_torch_obj(value: Any) -> bool:
    # TODO: verify this is faster than using `isinstance` or `torch.__name__ in getattr(value, '__module__', '')` (benchmark performances)
    mod = getattr(value, '_module__', None)
    if not mod:
        return False
    return torch.__name__ == mod.split('.')[0]


def is_fully_connected(module_or_t: MODULE_OR_TYPE_T) -> bool:
    return issubclass(module_or_t if isinstance(module_or_t, Type) else type(module_or_t), torch.nn.Linear)


def is_conv(module_or_t: MODULE_OR_TYPE_T) -> bool:
    """ Returns `True` if given `torch.nn.Module` instance or type is a convolution operation (i.e. inherits from `torch.nn.modules.conv._ConvNd`); Returns `False` otherwise. """
    return issubclass((module_or_t if isinstance(module_or_t, Type) else type(module_or_t)), torch.nn.modules.conv._ConvNd)


def contains_conv(module: torch.nn.Module) -> bool:
    """ Returns `True` if given `torch.nn.Module` contains at least one convolution module/op (based on `deepcv.meta.nn.is_conv` for convolution definition) """
    return any(map(module.modules, lambda m: is_conv(m)))

#____________________________________________________ UNIT TESTS ______________________________________________________#


class TestNNMetaModule:
    def test_is_conv(self):
        convs = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d, torch.nn.Conv2d(3, 16, (3, 3))]
        not_convs = [torch.nn.Linear, torch.nn.Linear(32, 32), tuple(), int, 54, torch.Tensor(), torch.nn.Fold, torch.nn.Conv2d(3, 16, (3, 3)).weight]
        assert all(map(is_conv, convs)), 'TEST ERROR: is_conv function failed to be true for at least one torch.nn convolution type or instance.'
        assert not any(map(is_conv, not_convs)), 'TEST ERROR: is_conv function failed to be false for at least one non-convolution type or instance.'

    def test_func_to_module(self):
        def _case1(): pass
        def _case2(param): assert param == 2
        def _case3(param1, param2=3): assert param1 == 3 and param2 == 3
        def _case4(param1: torch.Tensor, **kwparams): return kwparams

        _M1 = func_to_module('M1')(_case1)
        _M2 = func_to_module('M2')(_case2)
        _M3 = func_to_module('M3')(_case3)
        M4 = func_to_module('M4', ['truc', 'bidule'])(_case4)

        m4 = M4(truc='1', bidule=2)
        assert m4.forward(torch.zeros((16, 16))) == {'truc': '1', 'bidule': 2}

        @func_to_module('M5')
        def _case5(a: torch.Tensor): return a
        @func_to_module('M6')
        def _case6(param: str = 'test'): assert param == 'test'

        m6 = _case6()
        m6.forward()


if __name__ == '__main__':
    cli = import_tests().test_module_cli(__file__)
    cli()
