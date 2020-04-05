#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Neural Network meta module - nn.py - `DeepCV`__
Defines various neural network building blocks (layers, architectures parts, transforms, loss terms, ...)
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import inspect
import logging
import numpy as np
from enum import Enum, auto
from types import SimpleNamespace
from collections import OrderedDict
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence

import torch
import torch.nn as nn
from torch.functional import F
import torch.distributions as tdist
from hilbertcurve.hilbertcurve import HilbertCurve

from deepcv import utils
from ...tests.tests_utils import test_module
from torch import Tensor, squeeze, zeros

__all__ = ['DeepcvModule', 'HybridConnectivityGatedNet', 'Flatten', 'MultiHeadModule', 'ConcatHilbertCoords', 'func_to_module', 'layer', 'conv_layer', 'fc_layer',
           'resnet_net_block', 'squeeze_cell', 'multiscale_exitation_cell', 'meta_layer', 'concat_hilbert_coords_channel', 'flatten', 'get_gain_name',
           'parrallelize', 'mean_batch_loss', 'is_conv', 'contains_conv', 'contains_only_convs', 'parameter_summary']
__author__ = 'Paul-Emmanuel Sotir'


class DeepcvModule(nn.Module):
    HP_DEFAULTS = ...

    def __init__(self, input_shape: torch.Size, hp: Dict[str, Any]):
        super(DeepcvModule, self).__init__()
        assert HP_DEFAULTS != ..., f'Error: Module classes which inherits from "DeepcvModule" ({self.__class__.__name__}) must define "HP_DEFAULTS" class attribute dict.'

        self.input_shape = input_shape
        self.hyper_params = {n: v for n, v in hp if n in HP_DEFAULTS}
        self.hyper_params.update({n: v for n, v in HP_DEFAULTS if n not in hp and v != ...})
        missing_hyperparams = [n for n in HP_DEFAULTS if n not in self.hyper_params]
        assert len(missing_hyperparams) > 0, f'Error: Missing required hyper-parameter in "{self.__class__.__name__}" module parameters'


class HybridConnectivityGatedNet(DeepcvModule):
    """ Implementation of Hybrid Connectivity Gated Net (HCGN), residual/dense conv block architecture from the following paper: https://arxiv.org/pdf/1908.09699.pdf """
    HP_DEFAULTS = {'modules': ..., 'batch_norm': None, 'dropout_prob': 0.}

    def __init__(self, input_shape: torch.Size, hp: Dict[str, Any]):
        """ HybridConnectivityGatedNet __init__ function
        Args:
            hp: Hyperparameters
        """
        super(HybridConnectivityGatedNet, self).__init__(input_shape, hp)
        smg_modules = []
        for i, module_opts in enumerate(hp['modules']):
            prev_module = smg_modules[-1]
            gating = 'TODO'  # TODO: !!
            raise NotImplementedError
            ops = [('cell1', squeeze_cell(hp)), ('cell2', multiscale_exitation_cell(hp)), ('gating', gating)]
            smg_modules.append((f'smg_module_{i}', nn.Sequential(OrderedDict(ops))))
        self.net = nn.Sequential(OrderedDict(smg_modules))

    def forward(self, x: torch.Tensor):
        """ Forward propagation of given input tensor through conv neural network
        Args:
            - input: Input tensor fed to convolutional neural network (must be of shape (N, C, W, H))
        """
        return self.net(x)


def func_to_module(typename: str, init_params: Union[Sequence[str], Sequence[inspect.Parameter]] = []):
    """ Returns a decorator which creates a new ``torch.nn.Module``-based class using ``forward_func`` as underlying forward function.  
    Note: If ``init_params`` isn't empty, then returned ``nn.Module``-based class won't have the same signature as ``forward_func``.
    This is because some arguments provided to ``forward_func`` will instead be attributes of created module, taken by class's ``__init__`` function.  
    Args:
        - typename: Returned nn.Module class's ``__name__``
        - init_params: An iterable of string parameter name(s) of ``forward_func`` which should be taken by class's ``__init__`` function instead of ``forward`` function.
    """
    def _warper(forward_func: Callable) -> Type[nn.Module]:
        """ Returned decorator converting a function to a nn.Module class
        Args:
            - forward_func: Function from which nn.Module-based class is built. ``forward_func`` will be called on built module's ``forward`` function call.
        """
        signature = inspect.signature(forward_func)

        if len(init_params) > 0 and not type(init_params)[0] is inspect.Parameter:
            init_params = [signature.parameters[n] for n in init_params]
        forward_params = [p for p in signature if p not in init_params]
        init_signature = signature.replace(parameters=init_params, return_annotation=)
        forward_signature = signature.replace(parameters=forward_params)

        class _Module(nn.Module):
            def __init__(self, *args, **kwargs):
                super(_Module, self).__init__()
                bound_args = init_signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
                self.__dict__.update(bound_args.arguments)

            def forward(self, *inputs, **kwargs) -> torch.Tensor:
                bound_args = forward_signature.bind(*inputs, **kwargs)
                bound_args.apply_defaults()
                return forward_func(**bound_args.arguments, **self.__dict__)

        _Module.__init__.__annotations__ = init_signature.__annotations__
        _Module.__init__.__defaults__ = {n: p.default for (n, p) in init_signature.parameters.items()}
        _Module.forward.__annotations__ = forward_signature.__annotations__
        _Module.forward.__defaults__ = {n: p.default for (n, p) in forward_signature.parameters.items()}
        _Module.__name__ = typename
        return _Module
    return _warper


def flatten(x: torch.Tensor, from_dim: int = 0):
    """ Flattens tensor dimensions following ``from_dim``th dimension. """
    return x.view(*x.shape[:from_dim + 1], -1)


def multi_head_forward(x: torch.Tensor, embedding_shape: torch.Size, heads: Iterable[nn.Module]) -> torch.Tensor:
    new_siamese_dim = -len(embedding_shape)
    return torch.cat([head(x).unsqueeze(new_siamese_dim) for head in heads], dim=new_siamese_dim - 1)


def concat_hilbert_coords_channel(features: torch.Tensor, channel_dim: int = 0) -> torch.Tensor:
    """ Concatenates to feature maps a new channel which contains position information using Hilbert curve distance metric.
    This operation is close to CoordConv's except that we only append one channel of hilbert distance instead of N channels of euclidian coordinates (e.g. 2 channel for features from a 2D convolution).
    Args:
        - features: N-D Feature maps torch.Tensor with channel dimmension located at ``channel_dim``th dim and feature map dims located after channel's one. (Hilbert curve distance can be computed for any number, N, of feature map dimensions)
        - channel_dim: Channel dimension index, 0 by default.
    """
    assert features.dim() > 1, 'Invalid argument: "features" tensor should be at least of 2 dimensions.'
    assert channel_dim < features.dim() and channel_dim >= -features.dim(),
    'Invalid argument: "channel_dim" must be in [-features.dim() ; -1[ U ]-1 ; features.dim()[ range'

    if channel_dim < 0:
        channel_dim += features.dim()

    feature_map_size = features.shape[channel_dim + 1:]
    space_filling = HilbertCurve(n=len(feature_map_size), p=np.max(feature_map_size))

    space_fill_coords_map = np.zeros(feature_map_size)
    for coords in np.ndindex(feature_map_size):
        space_fill_coords_map[coords] = space_filling.distance_from_coordinates(coords)
    space_fill_coords_map = torch.from_numpy(space_fill_coords_map).view([1] * (channel_dim + 1) + [*feature_map_size])
    return torch.cat([space_fill_coords_map, features], dim=channel_dim)


# Torch modules created from their resective forward function:
Flatten = func_to_module('Flatten', ['from_dim'])(flatten)
MultiHeadModule = func_to_module('MultiHeadModule', ['embedding_shape', 'heads'])(multi_head_forward)
ConcatHilbertCoords = func_to_module('ConcatHilbertCoords', ['channel_dim'])(concat_hilbert_coords_channel)


def layer(layer_op: nn.Module, act_fn: nn.Module, dropout_prob: Optional[float] = None, batch_norm: Optional[dict] = None, preactivation: bool = False) -> Tuple[nn.Module]:
    """ Defines neural network layer operations
    Args:
        - layer_op: Layer operation to be used (e.g. nn.Conv2D, nn.Linear, ...).
        - act_fn: Activation function
        - dropout_prob: Dropout probability (if dropout_prob is None or 0., then no dropout ops is used)
        - batch_norm: (if batch_norm is None, then no batch norm is used)
        - preactivation: Boolean specifying whether to use preactivatation operation order: "(?dropout) - (?BN) - Act - Layer" or default operation order: "(?Dropout) - Layer - Act - (?BN)"
    Note:
        Note that dropout used along with batch norm may be unrecommended (see respective warning message).
    """
    assert 'weight' in layer_op.__dict__, f'Error: Bad layer operation module argument, no "weight" attribute found in layer_op="{layer_op}"'
    assert 'out_channels' in layer_op.__dict__ or 'out_features' in layer_op.__dict__, f'Error: Bad layer operation module argument, no "out_channels" or "out_features" attribute found in layer_op="{layer_op}"'

    def _dropout() -> Optional[nn.Module]:
        if dropout_prob is not None and dropout_prob != 0.:
            if batch_norm is not None:
                logging.warn(
                    "Warning: Dropout used along with batch norm may be unrecommended, see [CVPR 2019 paper: 'Understanding the Disharmony Between Dropout and Batch Normalization by Variance'](https://zpascal.net/cvpr2019/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf)")
            return nn.Dropout(p=dropout_prob)

    def _bn() -> Optional[nn.Module]:
        if batch_norm is not None:
            # Applies Batch_narm after activation function : see reddit thread about it : https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
            if layer_op.weight.dim() == 4:
                return nn.BatchNorm2d(layer_op.out_channels, **batch_norm)
            elif layer_op.weight.dim() < 4:
                return nn.BatchNorm1d(layer_op.out_features, **batch_norm)

    ops_order = (_dropout, _bn, act_fn, layer_op) if preactivation else (_dropout, layer_op, act_fn, _bn)
    ops = [op if issubclass(type(op), nn.Module) else op() for op in ops_order]
    return tuple(filter(lambda x: x is None, ops))


def conv_layer(conv2d: dict, act_fn: type = nn.Identity, dropout_prob: float = 0., batch_norm: Optional[dict] = None) -> nn.Module:
    return nn.Sequential(*layer(nn.Conv2d(**conv2d), act_fn(), dropout_prob, batch_norm))


def fc_layer(linear: dict, act_fn: type = nn.Identity, dropout_prob: float = 0., batch_norm: Optional[dict] = None) -> nn.Module:
    return nn.Sequential(*layer(nn.Linear(**linear), act_fn(), dropout_prob, batch_norm))


def resnet_net_block(hp: SimpleNamespace) -> nn.Module:
    raise NotImplementedError
    ops = [('conv1', conv_layer(**hp.conv2d)), ('conv2', conv_layer(**hp.conv2d))]
    return nn.Sequential(OrderedDict(ops))


def squeeze_cell(hp: SimpleNamespace) -> nn.Module:
    raise NotImplementedError


def multiscale_exitation_cell(hp: SimpleNamespace) -> nn.Module:
    raise NotImplementedError


img = torch.zeros((3, 100, 100))

conv1 = nn.ReLU(nn.Conv2d(3, 16, (3, 3), padding=(3, 3), padding_mode='zero'))

net = nn.Sequential(('conv1', conv1), ('meta_1_2', conv_with_meta), ('conv2', conv2))

meta_1_2 = meta_layer((16, ))

ouput = net(img)


def ConvWithMetaLayer(nn.Module):
    def __init__(self, preactivation: bool = False):
        self.meta = layer(layer_op=, act_fn=nn.ReLU, dropout_prob=0., batch_norm=None, preactivation=preactivation)
        self.conv = nn.ReLU(nn.Conv2d(16, 3, (3, 3)))  # TODO: preactivation, etc...
        self.RANDOM_PROJ = torch.randn_like(self.conv.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        meta_out = self.meta(x)
        weight_scale, meta_out = meta_out.split((1, meta_out.size(-1) - 1), dim=-1)
        scaled_w = torch.mul(self.conv.weights.data, weight_scale)
        meta_out = meta_out.reshape((1,) * (self.conv.weights.dim() - 1) + (-1,))
        meta_out = meta_out.prod(self.RANDOM_PROJ)
        self.conv.weights = torch.add(meta_out, scaled_w)
        return self.conv(x)


def meta_layer(input_feature_shape: torch.Size, target_module: nn.Parameter):
    """ A 'parrallel'/'meta' layer applied to previous layer/block's features to infer global statistics of next layer's weight matrix
    Args:
        - layer_op: Underlying layer operation module
    """
    underlying_layer_ops = layer(layer_op: nn.Module, act_fn: nn.Module, dropout_prob: Optional[float]=None, batch_norm: Optional[dict]=None, preactivation: bool=False)
    ops = [('underlying_layer_ops', underlying_layer_ops), ]

    return nn.Sequential(OrderedDict(ops))


def get_gain_name(act_fn: type) -> str:
    """ Intended to be used with nn.init.calculate_gain(str):
    .. Example: nn.init.calculate_gain(get_gain_act_name(nn.ReLU))
    """
    if act_fn is nn.ReLU:
        return 'relu'
    elif act_fn is nn.LeakyReLU:
        return 'leaky_relu'
    elif act_fn is nn.Tanh:
        return 'tanh'
    elif act_fn is nn.Identity:
        return 'linear'
    else:
        raise Exception("Unsupported activation function, can't initialize it.")


def parrallelize(model: nn.Module) -> nn.Module:
    """ Make use of all available GPU using nn.DataParallel
    NOTE: ensure to be using different random seeds for each process if you use techniques like data-augmentation or any other techniques which needs random numbers different for each steps. TODO: make sure this isn't already done by Pytorch?
    """
    if torch.cuda.device_count() > 1:
        print(f'> Using "nn.DataParallel(model)" on {torch.cuda.device_count()} GPUs.')
        model = nn.DataParallel(model)
    return model


def mean_batch_loss(loss: nn.loss._Loss, batch_loss: torch.Tensor, batch_size=1) -> Optional[utils.Number]:
    if loss.reduction == 'mean':
        return batch_loss.item()
    elif loss.reduction == 'sum':
        return torch.div(batch_loss, batch_size).item()
    elif loss.reduction == 'none':
        return torch.mean(batch_loss).item()


# class generic_mulltiscale_class_loss(nn.loss._Loss):
#     def __init__(self, reduction: str = 'mean') -> None:
#         self._terms = [nn.loss.MultiLabelMarginLoss, nn.loss.BCELoss, nn.loss.MultiLabelSoftMarginLoss, nn.loss.KLDivLoss]

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError

#     def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return self.forward(input, target)


# class generic_mulltiscale_class_reg_loss(nn.loss._Loss):
#     TERMS = [nn.loss.PoissonNLLLoss, nn.loss.SmoothL1Loss, nn.loss.MSELoss, nn.loss.HingeEmbeddingLoss, nn.loss.CosineEmbeddingLoss]

#     def __init__(self, reduction: str = 'mean', weights: torch.Tensor = torch.Tensor([1.] * len(TERMS))) -> None:
#         self._norm_factors = torch.Tensor([1. / len(TERMS))] * len(TERMS))
#         self._weights=weights
#         self._terms=[T(reduction= reduction) for T in TERMS]

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return
#         raise NotImplementedError

#     def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return self.forward(input, target)


def is_conv(op_t: Union[nn.Module, Type]) -> bool:
    from torch.nn.modules.conv import _ConvNd
    if not issubclass(type(op_t), Type):
        return issubclass(type(op_t), _ConvNd)
    return issubclass(op_t, _ConvNd)

# TODO: implement nn reflection tools (e.g. is_conv, contains_conv, parameter_summary, ...)


def contains_conv(op_t: Union[nn.Module, Type]) -> bool:
    raise NotImplementedError


def contains_only_convs(op_t: Union[nn.Module, Type]) -> bool:
    raise NotImplementedError


def parameter_summary(op_t: Union[nn.Module, Type], pprint: bool = False) -> dict:
    raise NotImplementedError


class TestNNMetaModule:
    def test_is_conv(self):
        convs = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Conv2d(3, 16, (3, 3))]
        not_convs = [nn.Linear, nn.Linear(32, 32), tuple(), int, 54, torch.Tensor(), nn.Fold, nn.Conv2d(3, 16, (3, 3)).weight]
        assert all(map(is_conv, convs)), 'TEST ERROR: is_conv function failed to be true for at least one torch.nn convolution type or instance.'
        assert not any(map(is_conv, not_convs)), 'TEST ERROR: is_conv function failed to be false for at least one non-convolution type or instance.'

    def test_func_to_module(self):
        def _case1(): pass
        def _case2(param): assert param == 2
        def _case3(param1, param2=3): assert param1 == 3 and param2 == 3
        def _case4(param1: torch.Tensor, **kwparams): return kwparams

        M1 = func_to_module('M1')(_case1)
        M2 = func_to_module('M2')(_case2)
        M3 = func_to_module('M3')(_case3)
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
    test_module(__file__)
