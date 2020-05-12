#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Neural Network meta module - nn.py - `DeepCV`__  
Defines various neural network building blocks (layers, architectures parts, transforms, loss terms, ...)
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
TODO: Add EvoNorm_B0 and EvoNorm_S0 layer implentations (from deepmind neural architecture search results for normalized-activation conv layers)
"""
import copy
import inspect
import logging
from enum import Enum, auto
from types import SimpleNamespace
from collections import OrderedDict
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence

import numpy as np

import torch
import torch.nn as nn
from torch.functional import F
import torch.distributions as tdist
from hilbertcurve.hilbertcurve import HilbertCurve

import deepcv.utils
import deepcv.meta.base_module

__all__ = ['HybridConnectivityGatedNet', 'Flatten', 'MultiHeadConcat', 'ConcatHilbertCoords', 'func_to_module', 'layer', 'conv_layer', 'fc_layer',
           'resnet_net_block', 'squeeze_cell', 'multiscale_exitation_cell', 'meta_layer', 'concat_hilbert_coords_channel', 'flatten', 'get_gain_name',
           'data_parallelize', 'is_data_parallelization_usefull_heuristic', 'mean_batch_loss', 'find_best_eval_batch_size', 'get_model_capacity',
           'get_out_features_shape', 'type_or_instance_is', 'is_fully_connected', 'is_conv', 'contains_conv', 'contains_only_convs', 'parameter_summary']
__author__ = 'Paul-Emmanuel Sotir'


class HybridConnectivityGatedNet(deepcv.meta.base_module.DeepcvModule):
    """ Implementation of Hybrid Connectivity Gated Net (HCGN), residual/dense conv block architecture from the following paper: https://arxiv.org/pdf/1908.09699.pdf """
    HP_DEFAULTS = {'architecture': ..., 'act_fn': nn.ReLU, 'batch_norm': None, 'dropout_prob': 0.}

    def __init__(self, input_shape: torch.Size, hp: Dict[str, Any]):
        """ HybridConnectivityGatedNet __init__ function
        Args:
            hp: Hyperparameters
        """
        super(HybridConnectivityGatedNet, self).__init__(input_shape, hp)
        submodule_creators = deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS.update({'smg_module': self._smg_module_creator})
        self._net = self._define_nn_architecture(hp['architecture'], submodule_creators)
        self._initialize_parameters(hp['act_fn'])

        # smg_modules = []
        # for i, module_opts in enumerate(hp['modules']):
        #     prev_module = smg_modules[-1]
        #     gating = 'TODO'  # TODO: !!
        #     raise NotImplementedError
        #     ops = [('cell1', squeeze_cell(hp)), ('cell2', multiscale_exitation_cell(hp)), ('gating', gating)]
        #     smg_modules.append((f'smg_module_{i}', nn.Sequential(OrderedDict(ops))))
        # self.net = nn.Sequential(OrderedDict(smg_modules))

    def forward(self, x: torch.Tensor):
        """ Forward propagation of given input tensor through conv hybrid gated neural network
        Args:
            - input: Input tensor fed to convolutional neural network (must be of shape (N, C, W, H))
        """
        return self._net(x)

    @staticmethod
    def _smg_module_creator():
        raise NotImplementedError


def to_multiscale_inputs_model(model: deepcv.meta.base_module.DeepcvModule, scales: int = 3, no_downscale_dims: Tuple[int] = tuple()):
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


def to_multiscale_outputs_model(model: deepcv.meta.base_module.DeepcvModule, scales: int = 3, no_downscale_dims: Tuple[int] = tuple()):
    """
    TODO: similar implementation than to_multiscale_inputs_model
    """
    raise NotImplementedError


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
        init_signature = signature.replace(parameters=init_params, return_annotation=nn.Module)
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
                return forward_func(**bound_args.arguments, **vars(self))

        _Module.__name__ = typename
        _Module.__doc__ = f'Module created at runtime from `{forward_func.__name__}` forward function.\nInitial forward function documentation:\n' + forward_func.__doc__
        init_signature.return_annotation = _Module
        _Module.__init__.__annotations__ = init_signature.__annotations__
        _Module.__init__.__defaults__ = {n: p.default for (n, p) in init_signature.parameters.items() if p.default}
        _Module.forward.__annotations__ = forward_signature.__annotations__
        _Module.forward.__defaults__ = {n: p.default for (n, p) in forward_signature.parameters.items() if p.default}
        _Module.forward.__doc__ = _Module.__doc__
        return _Module
    return _warper


def flatten(x: torch.Tensor, from_dim: int = 0):
    """ Flattens tensor dimensions following ``from_dim``th dimension. """
    return x.view(*x.shape[:from_dim + 1], -1)


def multi_head_forward(x: torch.Tensor, heads: Iterable[nn.Module], concat_dim: int = 1, new_dim: bool = False) -> torch.Tensor:
    """ Forwards `x` tensor throught multiple head modules: contenates each given head module's output over features first dimension or a new dimension
    Args:
        - x: input tensor to be forwarded through head modules
        - heads: Head module taking `x` tensor as input and which output is concatenated over other heads dimension. All head modules must have the same output shape in order to be concatenated into output tensor (except on first features/`embedding_shape` dimension if `new_dim` is `False`)
        - concat_dim: By default, equals to `1`, which means that output tensor will be a concanetation of head's outputs tensors over 2nd dimension (typically, after batch dimension)
        - new_dim: Whether create a new concatenation dim or not. (defaults to `False`). For example, if `x` tensor is a batch of images or convolution outputs with channel dim after batch dimension, then if `new_dim=False` head modules output is concatenated over channel dim, otherwise output tensors are concatenated over a new dimension.
    """
    return torch.cat([head(x).unsqueeze(concat_dim) if new_dim else head(x) for head in heads], dim=concat_dim)


def concat_hilbert_coords_channel(features: torch.Tensor, channel_dim: int = 0) -> torch.Tensor:
    """ Concatenates to feature maps a new channel which contains position information using Hilbert curve distance metric.
    This operation is close to CoordConv's except that we only append one channel of hilbert distance instead of N channels of euclidian coordinates (e.g. 2 channel for features from a 2D convolution).
    Args:
        - features: N-D Feature maps torch.Tensor with channel dimmension located at ``channel_dim``th dim and feature map dims located after channel's one. (Hilbert curve distance can be computed for any number, N, of feature map dimensions)
        - channel_dim: Channel dimension index, 0 by default.
    # TODO: cache hilbert curve to avoid to reprocess it too often
    """
    assert features.dim() > 1, 'Invalid argument: `features` tensor should be at least of 2 dimensions.'
    assert channel_dim < features.dim() and channel_dim >= -features.dim(), 'Invalid argument: `channel_dim` must be in [-features.dim() ; -1[ U ]-1 ; features.dim()[ range'

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
MultiHeadConcat = func_to_module('MultiHeadConcat', ['heads', 'concat_dim', 'new_dim'])(multi_head_forward)
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
    if 'weight' not in vars(layer_op):
        raise ValueError(f'Error: Bad layer operation module argument, no `weight` attribute found in layer_op="{layer_op}"')
    if 'out_channels' not in vars(layer_op) and 'out_features' not in vars(layer_op):
        raise ValueError(f'Error: Bad layer op module argument, no `out_channels` nor `out_features` attribute in `layer_op={layer_op}`')

    def _dropout() -> Optional[nn.Module]:
        if dropout_prob is not None and dropout_prob != 0.:
            if batch_norm is not None:
                logging.warn("""Warning: Dropout used along with batch norm may be unrecommended, see 
                                [CVPR 2019 paper: 'Understanding the Disharmony Between Dropout and Batch Normalization by Variance'](https://zpascal.net/cvpr2019/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf)""")
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


def conv_layer(conv2d: dict, act_fn: type = nn.Identity, dropout_prob: float = 0., batch_norm: Optional[dict] = None, preactivation: bool = False) -> nn.Module:
    return nn.Sequential(*layer(nn.Conv2d(**conv2d), act_fn(), dropout_prob, batch_norm))


def fc_layer(linear: dict, act_fn: type = nn.Identity, dropout_prob: float = 0., batch_norm: Optional[dict] = None, preactivation: bool = False) -> nn.Module:
    return nn.Sequential(*layer(nn.Linear(**linear), act_fn(), dropout_prob, batch_norm))


def resnet_net_block(hp: SimpleNamespace) -> nn.Module:
    raise NotImplementedError
    ops = [('conv1', conv_layer(**hp.conv2d)), ('conv2', conv_layer(**hp.conv2d))]
    return nn.Sequential(OrderedDict(ops))


def squeeze_cell(hp: SimpleNamespace) -> nn.Module:
    raise NotImplementedError


def multiscale_exitation_cell(hp: SimpleNamespace) -> nn.Module:
    raise NotImplementedError


class ConvWithMetaLayer(nn.Module):
    def __init__(self, preactivation: bool = False):
        raise NotImplementedError
        self.conv = nn.Conv2d(16, 3, (3, 3))  # TODO: preactivation, etc...
        self.meta = layer(layer_op=self.conv, act_fn=nn.ReLU, dropout_prob=0., batch_norm=None, preactivation=preactivation)
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
    """ A 'parallel'/'meta' layer applied to previous layer/block's features to infer global statistics of next layer's weight matrix
    Args:
        - layer_op: Underlying layer operation module
    """
    raise NotImplementedError
    conv = nn.Conv2d(16, 3, (3, 3))
    underlying_layer_ops = layer(layer_op=conv, act_fn=nn.ReLU, dropout_prob=None, batch_norm=None, preactivation=False)
    ops = [('underlying_layer_ops', underlying_layer_ops), ]

    return nn.Sequential(OrderedDict(ops))


def get_gain_name(act_fn: Type[nn.Module]) -> str:
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


def data_parallelize(model: nn.Module, print_msg: bool = True) -> nn.Module:
    """ Make use of all available GPU using nn.DataParallel if there are multiple GPUs available
    NOTE: ensure to be using different random seeds for each process if you use techniques like data-augmentation or any other techniques which needs random numbers different for each steps. TODO: make sure this isn't already done by Pytorch?
    """
    if torch.cuda.device_count() > 1:
        print(f'> Using "nn.DataParallel({model})" on {torch.cuda.device_count()} GPUs.')
        model = nn.DataParallel(model)
    return model


def is_data_parallelization_usefull_heuristic(model: nn.Module, batch_shape: torch.Size, print_msg: bool = True) -> bool:
    """ Returns whether if data parallelization could be helpfull in terms of performances using a heuristic from model capacity, GPU count, batch_size and dataset's shape
    Args:
        - model: Model to be trained (computes its parameters capacity)
        - batch_shape: Dataset's batches shape
    # TODO: perform a random/grid search to find out factors
    """
    ngpus = torch.cuda.device_count()
    capacity_factor, batch_factor, ngpus_factor = 1. / (1024 * 1024), 1. / (1024 * 512), 1. / 8.
    if ngpus <= 1:
        return False
    capacity_score = 0.5 * F.sigmoid(np.log10(capacity_factor * get_model_capacity(model) + 1.)) / 5.
    batch_score = 3. * F.sigmoid(np.log10(np.log10(batch_factor * np.prod(batch_shape) + 1.) + 1.)) / 5.
    gpus_score = 1.5 * F.sigmoid(np.log10(ngpus_factor * (ngpus - 1.) + 1.)) / 5.  # TODO: improve this heuristic score according to GPU bandwidth and FLOPs?
    heuristic = capacity_score + batch_score + gpus_score
    if print_msg:
        negation, lt_gt_op = ('not', '>') if heuristic > 0.5 else ('', '<')
        print(f'DataParallelization may {negation} be helpfull to improve training performances: heuristic {lt_gt_op} 0.5 (heuristic = capacity_score({capacity_score:.3f}) + batch_score({batch_score:.3f}) + gpus_score({gpus_score}) = {heuristic:.3f})')
    return heuristic > 0.5


def mean_batch_loss(loss: torch.nn.modules.loss._Loss, batch_loss: torch.Tensor, batch_size=1) -> Optional[deepcv.utils.Number]:
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

def find_best_eval_batch_size(input_shape: torch.Size, *other_data_shapes: Iterable[torch.Size], model: Optional[nn.Module], safety_margin: float = 0.75, trial_growth_factor: float = 1.2, device=deepcv.utils.get_device(), **model_kwargs):
    """ Finds largest `batch_size` which could fit in video memory without issues in an evaluation setup (no backprop).
    This function is usefull to estimate best evaluation `batch_size`. If `model` argument is given, then 
    This gives an estimate/advice of maximal `batch_size` in a non-distributed setup, assuming you are training given model on current device.
    Otherwise, in distributed setup, you could call `find_best_eval_batch_size` in each training nodes and either use different eval `batch_size` in each nodes or take minimal returned value.
    Args:
        - data_shape: Input data tensor shape which will be contained in minibatches during evaluation
        - other_data_shapes: Other data tensor shapes which would be in minibatches dunring evaluation (e.g. targets minibatch). However, this argument can safely be ignored if your other/target data shapes are far smaller than input data size.
        - model: Optional model to be evaluated on batches
        - safety_margin: Factor applied to maximal `batch_size` estimate to make sure there is a safety margin in video memory usage
        - trial_growth_factor: `batch_size` exponential growth factor for each `batch_size` trials. (tries greater and greater `batch_size`s until memory overflow occurs)
        - model_kwargs: keyword arguments needed when calling optional `torch.nn.module` model (passed during forward pass along with a minibatch of dummy tensors of `data_shape` shape as input data)
    """
    assert safety_margin < 1. and safety_margin > 0., 'Error: `margin` argument should be comprised between 0. and 1..'
    assert trial_growth_factor > 1., 'Error: `trial_growth_factor` argument should be greater than 1.'

    max_batch_size = 1
    while True:
        try:
            max_batch_size = int(max_batch_size ** trial_growth_factor)
            x = torch.zeros((max_batch_size, input_shape)).to(device)
            others = [torch.zeros((max_batch_size, shape)).to(device) for shape in other_data_shapes]
            if model:
                model = model.to(device).eval()
                y = model(x, **model_kwargs)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # Ran out of memory: Free up memory
                if x:
                    del x
                if others:
                    del others
                if y:
                    del y
                torch.cuda.empty_cache()  # TODO: make sure this is a good idea to do this?
            else:
                raise e

    return safety_margin * max_batch_size


def get_model_capacity(model: nn.Module):
    return sum([np.prod(param.shape) for name, param in model.parameters(recurse=True)])


def get_out_features_shape(input_shape: torch.Size, module: nn.Module, input_batches: bool = True) -> torch.Size:
    """ Performs a forward pass with a dummy input tensor to figure out module's output shape """
    with torch.no_grad():
        dummy_batch_x = torch.unsqueeze(torch.zeros(input_shape), dim=0) if input_batches else torch.zeros(input_shape)
        return module(dummy_batch_x).shape


def type_or_instance_is(module_or_t: Any, type_to_check: Type) -> bool:
    if not issubclass(type(module_or_t), Type):
        return issubclass(type(module_or_t), type_to_check)
    return issubclass(module_or_t, type_to_check)


def is_fully_connected(module_or_t: Union[nn.Module, Type]) -> bool:
    return type_or_instance_is(module_or_t, nn.Linear)


def is_conv(module_or_t: Union[nn.Module, Type]) -> bool:
    from torch.nn.modules.conv import _ConvNd
    return type_or_instance_is(module_or_t, _ConvNd)

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
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
