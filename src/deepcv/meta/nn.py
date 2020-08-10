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
from enum import Enum, auto
from types import SimpleNamespace, FunctionType
from collections import OrderedDict
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence, List, Hashable

import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
import torch.distributions as tdist
from hilbertcurve.hilbertcurve import HilbertCurve

from deepcv.utils import NL, NUMBER_T, get_str_repr, import_tests
from deepcv.meta.types_aliases import *

__all__ = ['XAVIER_INIT_SUPPORTED_ACT_FN', 'ConvWithMetaLayer', 'Flatten', 'MultiHeadConcat', 'ConcatHilbertCoords', 'ConcatCoords',
           'func_to_module', 'flatten', 'to_multiscale_inputs_model', 'to_multiscale_outputs_model', 'multi_head_forward', 'concat_hilbert_coords_map', 'concat_coords_maps',
           'get_padding_from_kernel', 'nd_support', 'conv_nd', 'conv_transpose_nd', 'batch_norm_nd', 'instance_norm_nd', 'layer_norm_with_mean_only_batch_norm', 'NormTechnique',
           'NORM_TECHNIQUES_MODULES', 'normalization_techniques', 'layer', 'hrnet_input_stem', 'ParallelConvolution', 'MultiresolutionFusion', 'HRNetv1RepresentationHead', 'HRNetv2RepresentationHead', 'HRNetv2pRepresentationHead',
           'resnet_net_block', 'squeeze_cell', 'multiscale_exitation_cell', 'ConvWithMetaLayer', 'meta_layer', 'get_gain_name', 'data_parallelize', 'is_data_parallelization_usefull_heuristic',
           'ensure_mean_batch_loss', 'get_model_capacity', 'get_out_features_shape', 'is_fully_connected', 'is_conv', 'contains_conv']
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
#         self._terms = [torch.nn.loss.MultiLabelMarginLoss, torch.nn.loss.BCELoss, torch.nn.loss.MultiLabelSoftMarginLoss, torch.nn.loss.KLDivLoss]

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError

#     def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return self.forward(input, target)


# class generic_mulltiscale_class_reg_loss(torch.nn.loss._Loss):
#     TERMS = [torch.nn.loss.PoissonNLLLoss, torch.nn.loss.SmoothL1Loss, torch.nn.loss.MSELoss, torch.nn.loss.HingeEmbeddingLoss, torch.nn.loss.CosineEmbeddingLoss]

#     def __init__(self, reduction: str = 'mean', weights: torch.Tensor = torch.Tensor([1.] * len(TERMS))) -> None:
#         self._norm_factors = torch.Tensor([1. / len(TERMS))] * len(TERMS))
#         self._weights=weights
#         self._terms=[T(reduction= reduction) for T in TERMS]

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return
#         raise NotImplementedError

#     def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return self.forward(input, target)


# class HybridConnectivityGatedNet(deepcv.meta.base_module.DeepcvModule):
#     """ Implementation of Hybrid Connectivity Gated Net (HCGN), residual/dense conv block architecture from the following paper: https://arxiv.org/pdf/1908.09699.pdf """
#     HP_DEFAULTS = {'architecture': ..., 'act_fn': torch.nn.ReLU, 'batch_norm': None, 'dropout_prob': 0.}

#     def __init__(self, input_shape: torch.Size, hp: Dict[str, Any]):
#         """ HybridConnectivityGatedNet __init__ function
#         Args:
#             hp: Hyperparameters
#         """
#         super(HybridConnectivityGatedNet, self).__init__(input_shape, hp)
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

    def _warper(forward_func: Callable, typename: str, init_params: Sequence[Union[str, inspect.Parameter]]) -> Type[torch.nn.Module]:
        """ Returned decorator converting a function to a torch.nn.Module class
        Args:
            - forward_func: Function from which torch.nn.Module-based class is built. ``forward_func`` will be called on built module's ``forward`` function call.
        """
        if forward_func.__kwdefaults__ is not None and forward_func.__kwdefaults__ != {}:
            raise ValueError('Error: `forward_func` argument of `deepcv.meta.nn.func_to_module._warper` function cannot have keyword defaults (`__kwdefaults__` not supported)')
        signature = inspect.signature(forward_func)
        init_params = [n if type(n) is inspect.Parameter else signature.parameters[n] for n in init_params]
        forward_params = [p for n, p in signature.parameters.items() if p not in init_params]
        init_signature = signature.replace(parameters=init_params, return_annotation=torch.nn.Module)
        forward_signature = signature.replace(parameters=forward_params, return_annotation=signature.return_annotation)

        class _Module(torch.nn.Module):
            """ torch.nn.Module generated at runtime from given forward function by `deepcv.meta.nn.func_to_module` tooling function. """

            __module__ = globals().get('__module__')

            def __init__(self, *args, **kwargs):
                super(_Module, self).__init__()
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
            # TODO: make sure this is usefful: `__annotations__` is already copied by `functools.warps` but in case __annotations__ isn't defined, there may still be annotations accessible from `inspect.signature`:
            _Module.forward.__annotations__ = {n: getattr(p, 'annotation', None) for (n, p) in forward_signature.parameters.items()}
        return _Module
    return functools.partial(_warper, typename=typename, init_params=init_params)


def flatten(x: torch.Tensor, from_dim: int = 0) -> torch.Tensor:
    """ Flattens tensor dimensions following ``from_dim``th dimension. """
    return x.view(*x.shape[:from_dim + 1], -1)


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


def multi_head_forward(x: torch.Tensor, heads: Iterable[torch.nn.Module], concat_dim: int = 1, new_dim: bool = True) -> torch.Tensor:
    """ Forwards `x` tensor throught multiple head modules: contenates each given head module's output over features first dimension or a new dimension
    Args:
        - x: input tensor to be forwarded through head modules
        - heads: Head module taking `x` tensor as input and which output is concatenated over other heads dimension. All head modules must have the same output shape in order to be concatenated into output tensor (except on first features/`embedding_shape` dimension if `new_dim` is `False`)
        - concat_dim: By default, equals to `1`, which means that output tensor will be a concanetation of head's outputs tensors over 2nd dimension (typically, after batch dimension)
        - new_dim: Whether create a new concatenation dim or not. (defaults to `True`). For example, if `x` tensor is a batch of images or convolution outputs with `concat_dim=1` (channel dim), then if `new_dim=False` head modules output is stacked over channel dimension, otherwise output tensors are concatenated over a new dimension.
    """
    if new_dim:
        return torch.cat([head(x) for head in heads], dim=concat_dim)
    else:
        return torch.stack([head(x) for head in heads], dim=concat_dim)


def concat_coords_maps(x: torch.Tensor, channel_dim: int = 1):
    """ Concats N new features maps of euclidian coordinates (1D, 2D, ..., ND coordinates if `x` has N dimensions after `channel_dim`'s dimension) into given `x` tensor. Coordinates are concatenated at `channel_dim` dimention of `x` tenseor
    Args:
        - x: Input tensor which have at least 1 dimension after `channel_dim`'s dimension
        - channel_dim: Channel dimension index in `x` tensor at which coordinates maps will be concatenated (0 by default). Supports negative dim index: `channel_dim` must be in `]x.dim(); -1[ U ]-1 ; x.dim()[` range.
    Returns a tensor which is the concatenation of `x` tensor with coordinates feature map(s) at `channel_dim` dimension, allowing, for ex., to append pixel location information explicitly into data proceesed in your model(s).
    .. See also `deepcv.meta.nn.concat_hilbert_coords_map` (or its respective module `deepcv.meta.nn.ConcatHilbertCoords`) which is simmilar to `concat_coords_maps` (or `ConcatCoords` module alternative) but will only concatenate one coordinates map of Hilbert Curve coordinates/distance, no matter how many dimensions `x` have (location information takes less memory space by using Hilbert space filling curve instead of euclidian coordinates).

    Bellow is an axample of generated euclidian coordinates maps in case of 2D features:
    ```
    # If `x` is of shape (N, C, H, W) and `channel_dim` is 1, then euclidian coordinates maps which will be concatenated to `x` will be:
        0, 0    1, 0    ...     W , 0

        0, 1    1, 1    ...     W, 1

        ...     ...     ...     ...

        0, H    1, H    ...     H, W
    ```

    """
    return _concat_coords_maps_impl(x, channel_dim=channel_dim, euclidian=True)


def concat_hilbert_coords_map(x: torch.Tensor, channel_dim: int = 1):
    """ Concatenates to feature maps a new channel which contains position information using Hilbert curve distance metric.
    This operation is close to CoordConv's except that we only append one channel of hilbert distance instead of N channels of euclidian coordinates (e.g. 2 channel for features from a 2D convolution).
    Args:
        - features: N-D Feature maps torch.Tensor with channel dimmension located at `channel_dim`th dim and feature map dims located after channel's one. (Hilbert curve distance can be computed for any number, N, of feature map dimensions)
        - channel_dim: Channel dimension index, 1 by default.
    # TODO: cache hilbert curve to avoid to reprocess it too often
    """
    return _concat_coords_maps_impl(x, channel_dim=channel_dim, euclidian=False)


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
Flatten = func_to_module('Flatten', ['from_dim'])(flatten)
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
    """ Helper function allowing easier support for N-D operations/modules, see example usage bellow for better understanding (e.g. `deepcv.meta.nn.nd_batchnorm`). """
    if dims not in _nd_types:
        available_ops = ', '.join([f'{dim}D: {op.__name__ if isinstance(op, Type) else str(op)}' for dim, op in _nd_types.items()])
        raise ValueError(f'Error: {"This operator/module" if _name is  None else _name} doesnt support operations on {dims}D features maps, available ops are: `nd_types="{available_ops}"`'
                         f'(No {dims}D type/callable entry in `nd_types` of `deepcv.meta.nn.nd_support{f"(_name={_name}, ...)" if _name is not None else ""}`).')
    return _nd_types[dims](*args, **kwargs)


""" N-D Convolution operator based on `torch.nn.Conv*d` for 1D, 2D and 3D support """
conv_nd = functools.partial(nd_support, nd_types={1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}, _name='ConvNd')
""" N-D Transposed Convolution operator based on `torch.nn.ConvTranspose*d` for 1D, 2D and 3D support """
conv_transpose_nd = functools.partial(nd_support, nd_types={1: torch.nn.ConvTranspose1d, 2: torch.nn.ConvTranspose2d, 3: torch.nn.ConvTranspose3d}, _name='ConvTransposeNd')
""" N-D Batch Normalization module based on `torch.nn.BatchNorm*d` for 1D, 2D and 3D support """
batch_norm_nd = functools.partial(nd_support, _nd_types={1: torch.nn.BatchNorm3d, 2: torch.nn.BatchNorm3d, 3: torch.nn.BatchNorm3d}, _name='BatchNormNd')
""" N-D Instance Normalization module (a.k.a Constrast Normalization) based on `torch.nn.InstanceNorm*d` for 1D, 2D and 3D support """
instance_norm_nd = functools.partial(nd_support, nd_types={1: torch.nn.InstanceNorm1d, 2: torch.nn.InstanceNorm2d, 3: torch.nn.InstanceNorm3d}, _name='InstanceNormNd')


def layer_norm_with_mean_only_batch_norm(input_shape: torch.Size, eps=1e-05, elementwise_affine: bool = True, momentum: float = 0.1, track_running_stats: bool = True) -> torch.nn.Sequential:
    """ LayerNorm used along with 'mean-only' BatchNorm, as described in [`LayerNorm` paper](https://arxiv.org/pdf/1602.07868.pdf) """
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
    LAYER_NORM_WITH_MEAN_ONLY_BATCH_NORM = r'ln_with_mean_bn'


NORM_TECHNIQUES_MODULES = {NormTechnique.BATCH_NORM: deepcv.meta.nn.batch_norm_nd, NormTechnique.LAYER_NORM: torch.nn.LayerNorm, NormTechnique.INSTANCE_NORM: deepcv.meta.nn.instance_norm_nd,
                           NormTechnique.GROUP_NORM: torch.nn.GroupNorm, NormTechnique.LOCAL_RESPONSE_NORM: torch.nn.LocalResponseNorm, NormTechnique.LAYER_NORM_WITH_MEAN_ONLY_BATCH_NORM: layer_norm_with_mean_only_batch_norm}


def normalization_techniques(norm_type: Union[NormTechnique, Sequence[NormTechnique]], norm_kwargs: Union[Sequence[Dict[str, Any]], Dict[str, Any]], input_shape: torch.Size = None, supported_norm_ops: NORM_TECHNIQUES_MODULES_T = NORM_TECHNIQUES_MODULES) -> List[torch.nn.Module]:
    """ Creates `torch.nn.Module` operations for one or more normalization technique(s) as specified in `norm_type` (see `NORM_TECHNIQUES_MODULES` enum) and `norm_kwargs` (Keywoard arguments dict(s) given to their respective normalization Module)
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

    Returns `torch.nn.Module`(s) for normalization operation(s) as described by `norm_type`(s) and `norm_kwargs`

    NOTE: If `input_shape` arg isn't `None`, `instance_norm_nd`/`batch_norm_nd`'s `num_features` and `dims` args, LayerNorm's `normalized_shape` arg, GroupNorm's `num_channels` arg and `layer_norm_with_mean_only_batch_norm`'s `input_shape` argument are automatically specified from given features shape and wont be needed in `norm_kwargs` (only need to give other args in `norm_kwargs` for underlying norm `torch.nn.Module` constructor)
    NOTE: You cant specify the same normalization technique multiple times and `norm_type` and `norm_kwargs` must have the same lenght if those are `Sequence`s (multiple normalization ops)
    NOTE: According to results from Switchable-Normalization (SN) 2018 paper (https://arxiv.org/pdf/1811.07727v1.pdf):
        Instance normalization are used more often in earlier layers, batch normalization is preferred in the middle and layer normalization is used in the last more often.
        Intuitively, smaller batch sizes lead to a preference towards layer normalization and instance normalization (or Group Norm which is in between).
        .. See also following blog post about DNN normalization techniques: https://medium.com/techspace-usict/normalization-techniques-in-deep-neural-networks-9121bf100d8 (Note above inspired from it)
    TODO: Eventually setup warnings in cases where norm strategies doesnt seems compatible together or redundant (e.g.: May not play well = shape of normalized features not sufficent, ..., redundant = instance norm + another more general normalization, or group norm and layer norm and cases depending on parameters like group norm with only 1 group <=> Layer norm, Goup norm with C groups <=> Instance Norm, ...)
    TODO: Implement unit testing for this function
    """
    if not (isinstance(norm_type, (NormTechnique, str)) and isinstance(norm_kwargs, Dict)) and not (isinstance(norm_type, Sequence) and isinstance(norm_kwargs, Sequence) and len(norm_type) == len(norm_kwargs)):
        raise TypeError('Error: `norm_type` and `norm_kwargs` must either both be a sequence of the same size or both only one normalization technique and one keyword argument dict; '
                        f'Got `norm_type(s)="{norm_type}"` and `norm_kwargs="{norm_kwargs}"`')
    if isinstance(norm_type, (NormTechnique, str)):
        norm_type, norm_kwargs = [norm_type, ], [norm_kwargs, ]
    if len(set(norm_type)) != len(norm_type):
        raise ValueError(f'Error: Cant use the same normalization technique mutiple times at once (duplicates forbiden in `norm_type` argument; Got `norm_type(s)="{norm_type}"`')

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


def layer(layer_op: torch.nn.Module, act_fn: Optional[Type[torch.nn.Module]], dropout_prob: float = None, preactivation: bool = False,
          norm_type: Sequence[NormTechnique] = None, norm_kwargs: Sequence[Dict[str, Any]] = None, input_shape: torch.Size = None, supported_norm_ops: NORM_TECHNIQUES_MODULES_T = None) -> torch.nn.Module:
    """ Defines neural network layer operations
    Args:
        - layer_op: Layer operation to be used (e.g. torch.nn.Conv2d, torch.nn.Linear, ...).
        - act_fn: Activation function (If `None`, then no activation function is used)
        - dropout_prob: Dropout probability (if dropout_prob is None or 0., then no dropout ops is used)
        - norm_ops: Optional normalization module(s)/op(s), like `torch.nn.BatchNorm*d` module
        - preactivation: Boolean specifying whether to use preactivatation operation order: "(?dropout) - (?BN) - Act - Layer" or default operation order: "(?Dropout) - Layer - Act - (?BN)"
        - norm_type: Only needed for normalization technique(s); See respective argument of `deepcv.meta.nn.normalization_techniques` for more details (If `None`, then `deepcv.meta.nn.normalization_techniques` won't be called)
        - norm_kwargs: Only needed for normalization technique(s); See respective argument of `deepcv.meta.nn.normalization_techniques` for more details (If `None`, then `deepcv.meta.nn.normalization_techniques` won't be called)
        - input_shape: Only usefull for normalization technique(s); See respective argument of `deepcv.meta.nn.normalization_techniques` for more details
        - supported_norm_ops: Only usefull for normalization technique(s); See respective argument of `deepcv.meta.nn.normalization_techniques` for more details. (If `None`, then defaults to `deepcv.meta.nn.normalization_techniques`'s default value)

    Returns layer operations as a tuple of `torch.nn.Modules`
    NOTE: Dropout used along with batch norm may be unrecommended (see respective warning message).
    TODO: allow instance norm, layer norm, group norm as alternatives to batch norm
    TODO: allow grouped convolution (support from PyTorch) to be applied on varying feature map dimensions (HRNet) and/or different kernels (PyramidalConv)
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
    if norm_type is not None and norm_kwargs is not None and len(norm_type) * len(norm_kwargs) > 0:
        if not preactivation:
            # Postactivation; Normalization is applied after convolution, so we need to find tensor shape by applying conv and act_fn ops to a mock/dummy tensor of shape `input_shape`
            dummy_in_tensor = torch.zeros(input_shape)
            input_shape = get_out_features_shape(input_shape, torch.nn.Sequential(layer_op, act_fn), use_minibatches=True)[1:]
        supported_norms = dict(supported_norm_ops=supported_norm_ops) if supported_norm_ops is not None else dict()
        norm_ops = normalization_techniques(norm_type, norm_kwargs, input_shape=input_shape, **supported_norms)

    # Return a `torch.nn.Sequential` of layer operations (ops order differs depends on `preactivation` bool value)
    ops = (_dropout(), *norm_ops, act_fn(), layer_op) if preactivation else (_dropout(), layer_op, act_fn(), *norm_ops)
    return torch.nn.Sequential(*(m for m in ops if m is not None))


def hrnet_input_stem(input_shape: torch.Size, submodule_params: Dict[str, Tuple[Sequence[Any], Any]], conv_count: int = 2, act_fn: Union[Sequence[Optional[Type]], Optional[Type]] = torch.nn.ReLU,
                     preactivation: Union[Sequence[bool], bool] = False, dropout_prob: float = None, norm_type: Sequence[NormTechnique] = None, norm_kwargs: Sequence[Dict[str, Any]] = None,
                     supported_norm_ops: NORM_TECHNIQUES_MODULES_T = None, channel_dim: int = 1) -> torch.nn.Module:
    """ Input stem block as described in [HRNet NN architecture](https://arxiv.org/abs/1908.07919).
    This input module consists of `conv_count` (2 by defaults) 3x3 2-stride convolutions. Hence, input stem decreases input resolution by a `1/2^N` factor (`1/4` by default).
    NOTE: `submodule_params` have default values for all convolution ops arguments, except for `out_channels` which is needed (filter count). Avoid providing `in_channels` as this is already known from `input_shape`.
    """
    assert conv_count > 0, f'Error in `hrnet_input_stem`, `conv_count` argument must be greater than 0, got `conv_count={conv_count}`'
    conv_defaults = dict(kernel_size=(3, 3), stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
    conv_defaults.update(submodule_params)
    if 'in_channels' not in submodule_params:
        conv_defaults['in_channels'] = input_shape[channel_dim]

    strided_convs = list()
    for i in range(conv_count):
        conv_op = torch.nn.Conv2d(**conv_defaults)
        layer_op = layer(layer_op=conv_op, act_fn=act_fn, dropout_prob=dropout_prob, preactivation=preactivation,
                         norm_type=norm_type, norm_kwargs=norm_kwargs, input_shape=input_shape[channel_dim:], supported_norm_ops=supported_norm_ops)
        strided_convs.append((f'stem{"_strided" if conv_defaults["stride"] > 1 else ""}_conv_{i}', layer_op))
    return torch.nn.Sequential(OrderedDict(strided_convs))


class ParallelConvolution(torch.nn.Module):
    """ParallelConvolution multi-resolution/parallel convolution module which is a generalization of multi-resolution convolution modules of [HRNet paper](https://arxiv.org/abs/1908.07919) and Pyramidal Convolutions as described in [PyConv paper](https://arxiv.org/pdf/2006.11538.pdf).
    This `torch.nn.Module` operation performs a grouped conv operation where each groups are regular/independant convolutions which may be applied on different feature map resolutions (like HRNet parallel streams/branches) and/or have different conv parameters, like different kernel sizes for each group (like PyConv does).
    Thus, this module can be interpreted as being multiple regular convolutions applied to different feature maps and performed in parrallel NN branches.
    NOTE: Chaining such grouped multi-res convs in a `torch.nn.Sequential` is a simple way to define a siamese NN of parrallel convolutions. Moreover, combined with `deepcv.meta.nn.MultiresolutionFusion` module, `ParallelConvolution` allows to define HRNet-like NN architecture, where information from each siamese branches of parallel multi-res convs can flow between those throught fusion modules. (Multi-Resolution fusion modules can be compared to 'non-grouped' multi-resolution convolution, see repsective doc and HRNet paper for more details)
    """

    def __init__(self, input_shape: Union[torch.Size, Sequence[torch.Size]], submodule_params: Dict[str, Tuple[Sequence[Any], Any]], act_fn: Union[Sequence[Optional[Type]], Optional[Type]] = torch.nn.ReLU,
                 preactivation: Union[Sequence[bool], bool] = False, dropout_prob: float = None, norm_type: Sequence[NormTechnique] = None, norm_kwargs: Sequence[Dict[str, Any]] = None,
                 supported_norm_ops: NORM_TECHNIQUES_MODULES_T = None, channel_dim: int = 1, raise_on_not_enough_in_tensors: bool = True):
        """ Multi-resolution/parallel convolution module which is a generalization of multi-resolution convolution modules of [HRNet paper](https://arxiv.org/abs/1908.07919) and Pyramidal Convolutions as described in [PyConv paper](https://arxiv.org/pdf/2006.11538.pdf).
        This `torch.nn.Module` operation performs a grouped conv operation where each groups are regular/independant convolutions which may be applied on different feature map resolutions (like HRNet parallel streams/branches) and/or have different conv parameters, like different kernel sizes for each group (like PyConv does).
        Args:
            - input_shape: Input tensors shape(s), with channel dim located at `channel_dim`th dim and with or without minibatch dimension. These tensor shapes should be shapes on which each parrallel/multi-res convolutions will be performed. This argument can also be used by eventual normalization technique(s) (`input_shape[channel_dim:]` passed to `normalization_techniques`).
            - submodule_params: Keyword arguments dict passed `deepcv.meta.nn.conv_nd`. Each argument can either be a sequence (List, Tuple, ...) of the same lenght as `input_shape` or a single value depending on if each multires-group convs have differents arg values or share the same argument value.
            - ... for other arguments, see corresponding arguments in `deepcv.meta.nn.layer` documentation.
                NOTE: Note that `act_fn` and `preactivation`, like arguments in `submodule_params`, can also be sequences if different values for each group/parallel convs are needed, but normalization parameters and `dropout_prob` will be shared across all group/parallel convs (no support for different `dropout_prob`, `norm_type`, `norm_kwargs` and `supported_norm_ops` args across parallel convs for now)
            - channel_dim: Channel dim index in provided `input_shape` tensor shapes. Dimensions after channel dim should be spatial features dims on which convolution is applied (e.g. two dimensions after channel dim for a 2D convoltion)
            - raise_on_not_enough_in_tensors: Boolean indicating whether to raise an exception if too few input tensors are given during forward passes (less than `len(input_shape)` input tensors given to forward method)

        NOTE: `kernel_size`, `padding` and `dilation` args in `submodule_params` must allways be specified as a sequence of integers (one for each dims, e.g. `kernel_size` cant be a single int, like `2` for a `2x2` kernel, otherwise, there could be ambiguities as conv args can be specified once or as a sequence of values for each parrallel conv groups/branches). I.e., their value must match this type: `Union[Sequence[Sequence[int]], Sequence[int]]` where inner-most sequence being of lenght `dims` and optional parent sequence being of the smae lenght as `input_shape` sequence. See also related error message in function code.
        NOTE: This is an helper function which actually is a special use case of `deepcv.meta.nn.layer` function; For more details about `deepcv.meta.nn.parallel_convolution` arguments, see `deepcv.meta.nn.layer` documentation.

        .. See also related `deepcv.meta.nn.MultiresolutionFusion` module which is also part of basic HRNet architecture building blocks.
        .. See also [`HigherHRNet` paper](https://arxiv.org/pdf/1908.10357.pdf).
        """
        if isinstance(input_shape, torch.Size):
            input_shape = [input_shape, ]
        elif len(input_shape) == 0:
            raise ValueError('Error in `deepcv.meta.nn.parallel_convolution`, `input_shape`(s) argument must be a sequence of `torch.Size` with at least one input tensor shape(s). '
                             f'NOTE: If you are performing a single conv on a single tensor shape/resolution, use `deepcv.meta.nn.layer` instead. Got `input_shape="{input_shape}"`')
        self.raise_on_not_enough_in_tensors = raise_on_not_enough_in_tensors
        self.channel_dim = channel_dim
        self.spatial_dims = len(input_shape[0][channel_dim+1:])

        # Share the same activation function and/or preactivation for all group/parallel convolution layers if those are not sequences
        if not isinstance(act_fn, Sequence):
            act_fn = [act_fn, ] * len(input_shape)
        if not isinstance(preactivation, Sequence):
            preactivation = [preactivation, ] * len(input_shape)
        # If act_fn and/or preactivation where sequences, those should have the same lenght as `input_shape` (one different value for each parallel convolution)
        if any([not len(input_shape) == len(seq) for seq in {act_fn, preactivation}]):
            raise ValueError('Error: If `preactivation` and/or `act_fn` args are sequences, those should have the same lenght as `input_shape` (one value for each parallel convolution)')

        # Forbid to specify a single int instead of a sequence of `dims` ints. Nescessary to avoid ambiguities between a sequence of different args for each group/parallel convs or a sequence of ints for each feature map dim (would be ambiguous when `dims==len(input_shape)`)
        conv_args_with_sequence_constraint = {'padding', 'kernel_size', 'dilation'}

        # Handle convs parameters to allow to either specify different parameters for each resolution(s)/group or a single parameter common to all mulit-res group convs.
        all_conv_ops_kwargs = list()
        for i in range(len(input_shape)):
            ith_conv_kwargs = dict(dims=self.spatial_dims)
            for n, v in submodule_params.items():
                if n in conv_args_with_sequence_constraint:
                    if isinstance(v, Sequence) and len(v) == len(input_shape) and all([isinstance(sub_v, Sequence) for sub_v in v]):
                        v = v[i]
                    elif not isinstance(v, Sequence) or len(v) not in {1, self.spatial_dims} or any([isinstance(sub_v, Sequence) for sub_v in v]):
                        raise ValueError(f'Error: `{n}` entry of `submodule_params` should either be a different sequence for each group/parallel conv (sequence of sequence of size `len(input_shape) x [dims={self.spatial_dims} or 1]`) or a single sequence of size `1` or `dims={self.spatial_dims}` (then the value is common to all group/parallel convs).{NL}'
                                         f'This is a needed constraint to avoid ambiguous spec in `parallel_convolution`; '
                                         f'E.g., in case you need a `3x2` kernel size for two parallel/group convs you can specify `kernel_size = [3, 2]` but in order to have two parallel/group convs with different kernel sizes of 3x3 and 2x2, then you need to specify: `kernel_size = [[3, 3], [2, 2]]` or alternatively: `[[3], [2]]`.{NL}'
                                         f'I.e. Unlike for regular usage of `torch.nn.Conv*d`, sequence of int(s) are allways needed for "{conv_args_with_sequence_constraint}" conv(s) args; Spec would be ambiguous otherwise')
                    ith_conv_kwargs[n] = v if len(v) > 1 else v[0]
                else:
                    ith_conv_kwargs[n] = v[i] if isinstance(v, Sequence) and len(v) == len(input_shape) else v
            all_conv_ops_kwargs.append(ith_conv_kwargs)

        self.group_convolutions = list()
        for i, (shape, ith_conv_kwargs, act, preact) in enumerate(zip(input_shape, all_conv_ops_kwargs, act_fn, preactivation)):
            conv_op = conv_nd(**ith_conv_kwargs)
            layer_op = layer(layer_op=conv_op, act_fn=act, dropout_prob=dropout_prob, preactivation=preact,
                             norm_type=norm_type, norm_kwargs=norm_kwargs, input_shape=shape[channel_dim:], supported_norm_ops=supported_norm_ops)
            self.group_convolutions.append((f'parallel_conv_{self.spatial_dims}D_{i}', layer_op))

    def forward(self, inputs: TENSOR_OR_SEQ_OF_TENSORS) -> TENSOR_OR_SEQ_OF_TENSORS:
        # Check if input tensors are valid
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs,]
        if len(inputs) > len(self.group_convolutions) or (self.raise_on_not_enough_in_tensors and len(inputs) < len(self.group_convolutions)):
            raise ValueError(f'Error: Cant apply `{type(self).__name__}` module with {len(self.group_convolutions)} parallel/group convolutions on {len(inputs)} input tensors.')
        # Append minibatch dim to input tensor(s) if it is missing
        inputs = [torch.unsqueeeze(x, dim=0) if len(x.shape) - self.spatial_dims < self.channel_dim + 1 else x for x in inputs]
        # Apply parallel/group convolutions
        output = list([parallel_conv(in_tensor) for in_tensor, parallel_conv in zip(inputs, self.group_convolutions)])
        return output if len(output) > 1 else output

    
class MultiresolutionFusion(torch.nn.Module):
    """ Multi-resolution Fusion module as described in [HRNet paper](https://arxiv.org/abs/1908.07919) NN architecture.
    This fusion module can be compared to a regular convolution layer but applied on feature maps with varying resolutions. However, in order to be applied in a 'fully connected' conv. way, each feature maps will be up/down-scaled to each target resolutions (either using bilinear upsampling followed by a 1x1 conv or by applying a 3x3 conv with stride of 2).
    Multi-resolution Fusion modules thus 'fuses' information across all siamese branches (i.e. across all resolutions) of HRNet architecture.

    .. See also related `deepcv.meta.nn.parallel_convolution` module which is also part of basic HRNet architecture building blocks.
    .. See also [`HigherHRNet` paper](https://arxiv.org/pdf/1908.10357.pdf).
    """

    def __init__(self, input_shape: Union[torch.Size, Sequence[torch.Size]], upscale_conv_kwargs: Dict[str, Any] = None, downscale_conv_kwargs: Dict[str, Any] = None, create_new_branch: bool = True, new_branch_channels: int = None, channel_dim: int = 1, reuse_scaling_convs: bool = True):
        """ Instanciate `MultiresolutionFusion` module which 'fuses' information across all multi-resoution siamese branches/streams, like described in [HRNet architecture paper](https://arxiv.org/abs/1908.07919)
        Args:
            - input_shape: Input tensors shapes for each parallel/siamese input branches/streams, with minibatch dim (change `channel_dim` accordingly if needed). All `input_shape` tensor shapes should have the same number of dimensions after `channel_dim` dimension.
            - upscale_conv_kwargs:
            - downscale_conv_kwargs:
            - create_new_branch: Boolean indicating whether to create a new stage by outputing (new parallel branch/stream) by outputing an additional tensor with lower resolution feature maps (See HRNet paper for more details on multiresolution fusion module stages)
            - new_branch_channels: If `create_new_branch`, you can specify how many channels/feature-maps are outputed for this new parallel branch/stream. If `None` (Default), then output the same sumber of channels as previous/upper branch (same kernel/filter count).
            - channel_dim: Channel dim index in provided `input_shape` tensor shapes. Dimensions after channel dim should be spatial features dims on which convolutions are applied (e.g. `input_shape` should have two dims after `channel_dim` for 2D convoltions)
            - reuse_scaling_convs: If `self.reuse_scaling_convs` is `True`, then all up/down scaling convolutions are reused when possible (i.e. weight sharing when (in_channels, out_channels) tuple is the same)

        NOTE: Avoid providing custom `in_channels`, `out_channels` nor `padding` through `upscale_conv_kwargs` and/or `downscale_conv_kwargs` as, by default, those are proccessed automatically from input_shapes and kernel sizes (may be invalid otherwise as `out_channels` must be similar to `in_channel` for a given stream/branch and padding better should keep feature maps size unchanged).
        NOTE: When `self.reuse_scaling_convs` is `True`, having similar number of filters/channels/feature-maps across parallel branches will allow much more weight/parameter sharing than if they are different between parallel streams/branches.
        NOTE: If not excpilitly provided in `downscale_conv_kwargs` and/or `upscale_conv_kwargs`, zero padding is used and padding values are proceessed from kernel sizes in order to obtain unchanged output feature map sizes (See `deepcv.meta.nn.get_padding_from_kernel`).
        """
        if isinstance(input_shape, torch.Size):
            input_shape = [input_shape, ]
        elif len(input_shape) == 0:
            raise ValueError('Error in `deepcv.meta.nn.MultiresolutionFusion`, `input_shape`(s) argument must be a sequence of `torch.Size` with at least one input tensor shape(s). '
                             f'NOTE: If you are performing a single conv on a single tensor shape/resolution, use `deepcv.meta.nn.layer` instead. Got `input_shape="{input_shape}"`')

        # `create_new_branch` indicates whether to create a new conv parallel branch/stream at a lower resoltion (See HRNet paper for more details on multiresolution fusion module)
        self.create_new_branch = create_new_branch
        self.input_shape = input_shape
        self.channel_dim = channel_dim
        self.reuse_scaling_convs = reuse_scaling_convs
        self.spatial_dims = len(input_shape[0][self.channel_dim+1:])  # Assume all input features have the same spatial dims count

        # Apply default up/down-scaling convolutions ops parameters and process padding from kernel size if needed
        upscale_conv_kwargs = dict(kernel_size=1, dims=self.spatial_dims).update(upscale_conv_kwargs)
        downscale_conv_kwargs = dict(kernel_size=3, stride=2, padding_mode='zero', dims=self.spatial_dims).update(downscale_conv_kwargs)
        for args in [upscale_conv_kwargs, downscale_conv_kwargs]:
            if 'padding' not in args:
                args['padding'] = get_padding_from_kernel(args['kernel_size'], warn_on_uneven_kernel=True)

        # New branch either outputs `new_branch_channels` (if provided), latest/upper branch `out_channels` (if provided) or latest/upper branch `in_channels` features-maps/chanels
        self.new_branch_channels = downscale_conv_kwargs.get('out_channels', input_shape[-1][channel_dim]) if new_branch_channels is None else new_branch_channels
        in_channels = [in_shape[channel_dim] for in_shape in input_shape]
        out_channels = [*in_channels, self.new_branch_channels] if self.create_new_branch else in_channels

        # Define all needed convolutions modules for up/down-scaling between input/output parallel streams/branches
        if self.reuse_scaling_convs:
            # If `self.reuse_scaling_convs` is True, then all up/down scaling convolutions are reused when possible (i.e. weight sharing when (in_channels, out_channels) tuple is the same)
            self.downscaling_3x3_convs = {(in_ch, out_ch): conv_nd(out_channels=out_ch, in_channels=in_ch, **downscale_conv_kwargs)
                                          for in_ch in in_channels for out_ch in out_channels}
            self.upscaling_1x1_convs = {(in_ch, out_ch): conv_nd(out_channels=out_ch, in_channels=in_ch, **upscale_conv_kwargs)
                                        for in_ch in in_channels for out_ch in out_channels}
        else:
            self.upscaling_1x1_convs = [[conv_nd(out_channels=out_ch, in_channels=in_ch, **upscale_conv_kwargs) for in_ch in in_channels] for out_ch in out_channels]
            self.first_downscaling_3x3_convs = [[conv_nd(out_channels=out_ch, in_channels=in_ch, **downscale_conv_kwargs) for in_ch in in_channels] for out_ch in out_channels]
            # We still 'reuse' some convolutions: Additional downsampling convs are used when downscaling by a factor of `stride^2` (4 by default) or more: One more 3x3 conv for each target branches, with `in_channels==out_channels`, is used in addition to the first convs in `first_downscaling_3x3_convs` which are different for each input branches due to different in-channels count (`in_channels` may be different to `out_channels` for the first downscaling convs applied)
            self.additional_downscaling_3x3_convs = [conv_nd(out_channels=out_ch, in_channels=out_ch, **downscale_conv_kwargs) for out_ch in out_channels]

    def _upsample(self, x: torch.Tensor, target_shape: torch.Size, in_branch_idx: int, out_branch_idx: int, align_corners: bool = False) -> torch.Tensor:
        """ Upscaling is performed by a bilinear upsampling followed by a 1x1 convolution to match target channel count """
        # Upscale input tensor `x` using (bi/tri)linear interpolation (or 'nearest' interpolation for tensor with 4D or more features maps)
        scale_mode = 'linear' if self.spatial_dims == 1 else ('bilinear' if self.spatial_dims == 2 else ('trilinear' if self.spatial_dims == 3 else 'nearest'))
        x = torch.nn.functional.interpolate(x, size=target_shape[self.channel_dim+1:], mode=scale_mode, align_corners=align_corners)

        # Apply 1x1 convolution in order to obtain the target channel/feature-maps count
        if self.reuse_scaling_convs:
            x = self.upscaling_1x1_convs[(x.shape[self.channel_dim], target_shape[self.channel_dim])](x)
        else:
            x = self.upscaling_1x1_convs[out_branch_idx][in_branch_idx](x)
        return x

    def _downsample(self, x: torch.Tensor, out_channels: int, apply_n_times: int, in_branch_idx: int, out_branch_idx: int) -> torch.Tensor:
        """ Downsampling is performed by applying N times a 3x3 convolution with a stride of 2 (by default)
        Each parallel branch/streams process feature maps at a resolution divided by 2, i.e. 4 times less features per map on 2D maps.
        """
        if apply_n_times <= 0:
            raise ValueError(f'Error: received bad argument value, `apply_n_times={apply_n_times}`, in `MultiresolutionFusion._downsample`')

        if self.reuse_scaling_convs:
            for i in range(apply_n_times):
                x = self.downscaling_3x3_convs[(x.shape[1], out_channels)](x)
        else:
            x = self.first_downscaling_3x3_convs[out_branch_idx][in_branch_idx](x)
            for _i in range(apply_n_times - 1):
                x = self.additional_downscaling_3x3_convs[out_branch_idx](x)
        return x

    def forward(self, inputs: TENSOR_OR_SEQ_OF_TENSORS) -> List[torch.Tensor]:
        """ Fusion is performed by summing all down/up-scaled input features from each streams/branches.
        Thus, all inputs are down/up-scaled to all other resolutions before being sumed (plus down-scaled to new branch/stream resolution if `create_new_branch`).
        """
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs,]
        if len(inputs) != len(self.input_shape):
            raise ValueError(f'Error: Cant apply `{type(self).__name__}` module with {len(self.input_shape)} input parallel branches/streams on {len(inputs)} input tensors.')

        # Append minibatch dim to input tensor(s) if it is missing and keep track of their respective usage of minibatch dim in order to restore it at module output
        had_minibatch_dim = [len(x.shape) - self.spatial_dims >= self.channel_dim + 1 for x in inputs]
        inputs = [x if has_minibatch_dim else torch.unsqueeeze(x, dim=0) for has_minibatch_dim, x in zip(had_minibatch_dim, inputs)]
        in_resolutions = [np.prod(x.shape[self.channel_dim+1:]) for x in inputs]
        output_shapes = [x.shape for x in ([*inputs, None] if self.create_new_branch else inputs)]

        def _get_downscale_n_times(input_branch_idx: int, out_branch_idx: int) -> int:
            """ Used to get how many times we need to downscale features from `input_branch_idx`th branch to `out_branch_idx`th by using sorted indices of input resolutions (inputs may not be sorted by resolution) """
            return np.where(_get_downscale_n_times.sorted_idx == out_branch_idx) - np.where(_get_downscale_n_times.sorted_idx == input_branch_idx)
        _get_downscale_n_times.sorted_idx = np.argsort(in_resolutions)
        if self.create_new_branch:
            # If we need to create a new branch, then we append its index at the end of sorted indices in order to obtain an additional downscaling
            _get_downscale_n_times.sorted_idx.append(len(in_resolutions))

        outputs = list()
        for out_branch_idx, target_shape in enumerate(output_shapes):
            scaled_features = []
            target_res = None if target_shape is None else np.prod(target_shape[self.channel_dim+1:])
            for in_branch_idx, (other_inputs, in_res) in enumerate(zip(inputs, in_resolutions)):
                if target_shape is None or in_res > target_res:
                    downscale_n_times = _get_downscale_n_times(in_branch_idx, out_branch_idx)
                    out_channels = self.new_branch_channels if out_branch_idx == len(inputs) else target_shape[self.channel_dim]
                    scaled_features.append(self._downsample(other_inputs, apply_n_times=downscale_n_times,
                                                            out_channels=out_channels, out_branch_idx=out_branch_idx, in_branch_idx=in_branch_idx))
                elif in_res < target_res:
                    scaled_features.append(self._upsample(other_inputs, target_shape=target_shape, out_branch_idx=out_branch_idx, in_branch_idx=in_branch_idx))
                else:
                    scaled_features.append(other_inputs)

            outputs.append(torch.sum(scaled_features))

        return [x if keep_minibatch else x.squeeze(dim=0) for keep_minibatch, x in zip(had_minibatch_dim, outputs)]


class HRNetv1RepresentationHead(torch.nn.Module):
    def __init__(self, input_shape: Union[torch.Size, Sequence[torch.Size]], channel_dim: int = 1) -> torch.nn.Module:
        if isinstance(input_shape, torch.Size):
            input_shape = [input_shape, ]
        self.input_shape = input_shape
        self.channel_dim = channel_dim

    def _forward(self, inputs: TENSOR_OR_SEQ_OF_TENSORS) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs,]
        if len(inputs) != len(self.input_shape):
            raise ValueError(f'Error: Cant apply `{type(self).__name__}` module with {len(self.input_shape)} input parallel branches/streams on {len(inputs)} input tensors.')
        max_idx = np.argmax(np.prod(x.shape[self.channel_dim+1:] for x in inputs))
        return inputs[max_idx]


class HRNetv2RepresentationHead(torch.nn.Module):
    def __init__(self, input_shape: Union[torch.Size, Sequence[torch.Size]], upscale_conv_kwargs: Dict[str, Any] = None, channel_dim: int = 1) -> torch.nn.Module:
        if isinstance(input_shape, torch.Size):
            input_shape = [input_shape, ]
        self.input_shape = input_shape
        self.channel_dim = channel_dim
        # TODO: ...    
        
    def forward(self, inputs: TENSOR_OR_SEQ_OF_TENSORS) -> TENSOR_OR_SEQ_OF_TENSORS:
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs,]
        if len(inputs) != len(self.input_shape):
            raise ValueError(f'Error: Cant apply `{type(self).__name__}` module with {len(self.input_shape)} input parallel branches/streams on {len(inputs)} input tensors.')
        # TODO: upscale input features from lwer branches...
    

class HRNetv2pRepresentationHead(HRNetv2RepresentationHead):
    def __init__(self, input_shape: Union[torch.Size, Sequence[torch.Size]], upscale_conv_kwargs: Dict[str, Any] = None, downscale_conv_kwargs: Dict[str, Any] = None, channel_dim: int = 1) -> torch.nn.Module:
        super().__init__(input_shape=input_shape, upscale_conv_kwargs=upscale_conv_kwargs, channel_dim=channel_dim)
        # TODO: ...
    
    def forward(self, inputs: TENSOR_OR_SEQ_OF_TENSORS) -> TENSOR_OR_SEQ_OF_TENSORS:
        hrnetv2_out = super().forward(inputs)
        # TODO: Downscale representation from HRNetV2 head to all input_shapes

def resnet_net_block(hp: Union[Dict[str, Any], 'deepcv.meta.hyperparams.Hyperparameters']) -> torch.nn.Module:
    raise NotImplementedError
    ops = [('conv1', layer(**hp['conv'])), ('conv2', layer(**hp['conv']))]
    return torch.nn.Sequential(OrderedDict(ops))


def squeeze_cell(hp: SimpleNamespace) -> torch.nn.Module:
    raise NotImplementedError


def multiscale_exitation_cell(hp: SimpleNamespace) -> torch.nn.Module:
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
        torch.nn.init.xavier_normal(weights_to_init, gain=gain) # You can alternatively use `torch.nn.init.xavier_uniform` according to your needs (e.g. linear layer weights init instead of convolution's)
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


def get_model_capacity(model: Optional[torch.nn.Module]):
    if model is None:
        return 0
    return sum([np.prod(param.shape) for param in model.parameters(recurse=True)])


def get_out_features_shape(input_shape: torch.Size, module: torch.nn.Module, use_minibatches: bool = True) -> Union[torch.Size, List[torch.Size], Dict[Hashable, torch.Size]]:
    """ Performs a forward pass with a dummy input tensor to figure out module's output shape.
    NOTE: `input_shape` is assumed to be input tensor shape without eventual minibatch dim: If `use_minibatches` is `True`, input tensor will be unsueezed to have a minibatch dim, along with `input_shape` dims, before being forwarded throught given `module`.
    Returns output tensor shape(s) of given `module` applied to a dummy input (`torch.nn.zeros`) with or without additional minibatch dim (depending on `use_minibatches`). If `module` returns a `Sequence` or a `Dict` of `torch.Tensor` instead of a single tensor, this function will return a list or a dict of output tensors shapes
    """
    module.eval()
    with torch.no_grad():
        dummy_batch_x = torch.unsqueeze(torch.zeros(input_shape), dim=0) if use_minibatches else torch.zeros(input_shape)
        outputs = module(dummy_batch_x)
        if isinstance(outputs, torch.Tensor):
            return outputs.shape
        return {n: r.shape for n, r in outputs.items()} if isinstance(outputs, Dict) else [r.shape for r in outputs]


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

        @ func_to_module('M5')
        def _case5(a: torch.Tensor): return a
        @ func_to_module('M6')
        def _case6(param: str = 'test'): assert param == 'test'

        m6 = _case6()
        m6.forward()


if __name__ == '__main__':
    cli = import_tests().test_module_cli(__file__)
    cli()
