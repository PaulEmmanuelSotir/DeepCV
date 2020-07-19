#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" DeepCV model base class meta module - base_module.py - `DeepCV`__
Defines DeepCV model base class
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
    - TODO: optimize support for residual/dense links
    - TODO: Try to unfreeze batch_norm parameters of shared image embedding block (with its other parameters freezed) and compare performances across various tasks
"""
import copy
import types
import inspect
import logging
from pathlib import Path
from functools import partial
from collections import OrderedDict
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence, List, Set

import torch
import torch.nn.functional as F

import numpy as np
import nni
import nni.nas.pytorch.mutables as nni_mutables

import deepcv.utils
import deepcv.meta.nn
import deepcv.meta.hyperparams


__all__ = ['MODULE_CREATOR_CALLBACK_RETURN_T', 'REDUCTION_FUNCTION_T', 'BASIC_SUBMODULE_CREATORS', 'TENSOR_REDUCTION_FUNCTIONS',
           'DeepcvModule', 'DeepcvModuleWithSharedImageBlock', 'DeepcvModuleDescriptor']
__author__ = 'Paul-Emmanuel Sotir'

MODULE_CREATOR_CALLBACK_RETURN_T = Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]
REDUCTION_FUNCTION_T = Callable[[List[torch.Tensor, 'dim'], Union[torch.Tensor, List[torch.Tensor]]]]
TENSOR_REDUCTION_FUNCTIONS = {'mean': torch.mean, 'sum': torch.sum, 'concat': torch.cat, 'none': lambda l, dim: l}


def _create_avg_pooling(submodule_params: Dict[str, Any], prev_shapes: List[torch.Size]) -> torch.nn.Module:
    prev_dim = len(prev_shapes[-1][1:])
    if prev_dim >= 4:
        return torch.nn.AvgPool3d(**submodule_params)
    elif prev_dim >= 2:
        return torch.nn.AvgPool2d(**submodule_params)
    return torch.nn.AvgPool1d(**submodule_params)


def _create_nn_layer(is_fully_connected: bool) -> Callable[['submodule_params', 'prev_shapes', int], torch.nn.Module]:
    """ Creates a fully connected or convolutional NN layer with optional dropout and batch norm support
    NOTE: We assume here that features/inputs are given in batches and that input only comes from previous sub-module (e.g. no direct residual/dense link)
    """
    def _create_conv_or_fc_layer(submodule_params: Dict[str, Any], prev_shapes: List[torch.Size], is_fully_connected: bool, act_fn: Optional[torch.nn.Module] = None, dropout_prob: Optional[float] = None, batch_norm: Optional[Dict[str, Any]] = None, channel_dim: int = 1) -> torch.nn.Module:
        if is_fully_connected:
            submodule_params['in_features'] = np.prod(prev_shapes[-1][1:])
            layer_nn_fn = deepcv.meta.nn.fc_layer
        else:  # Convolution layer
            submodule_params['in_channels'] = prev_shapes[-1][channel_dim]
            layer_nn_fn = deepcv.meta.nn.conv_layer
        return layer_nn_fn(submodule_params, act_fn, dropout_prob, batch_norm)

    _create_conv_or_fc_layer.__doc__ = _create_nn_layer.__doc__
    return partial(_create_conv_or_fc_layer, is_fully_connected=is_fully_connected)


def _residual_dense_link(is_residual: bool = True) -> Callable[['submodule_params', 'prev_named_submodules', 'prev_submodules', int], MODULE_CREATOR_CALLBACK_RETURN_T]:
    """ Creates a residual or dense link sub-module which concatenates or adds features from previous sub-module output with other referenced sub-module(s) output(s).
    `submodule_params` argument must contain a `_from` (or `_from_nni_mutable_input`) entry giving the other sub-module reference(s) (sub-module name(s)) from which output(s) is added to previous sub-module output features.

    Returns a callback which is called during foward pass of DeepcvModule.
    Like any other DeepcvModule submodule creators which returns a forward callback and which uses tensor references (`_from` or `_from_nni_mutable_input`), a reduction function can be specified (see `REDUCTION_FUNCTION_T`) in `_from` or `_from_nni_mutable_input` parameters: by default, reduction function will be 'sum' if this is a residual link and 'concat' if this is a dense link. 
    If 'allow_scaling' is `True` (when `allow_scaling: True` is specified in YAML residual/dense link spec.), then if residual/dense features have different shapes on dimensions following `channel_dim`, it will be scaled (upscaled or downscaled) using [`torch.functional.interpolate`](https://pytorch.org/docs/stable/nn.functional.html#interpolate).
    By default interpolation is 'linear'; If needed, you can specify a `scaling_mode` parameter as well to change algorithm used for up/downsampling (valid values: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area', for more details, see [`torch.nn.functional.interpolate` doc](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate)).
    NOTE: Scaling/interpolation of residual/dense tensors is only supported for 1D, 2D and 3D features, without taking into account channel and minibatch dimensions. Also note that `minibatch` dimension is required by [`torch.functional.interpolate`](https://pytorch.org/docs/stable/nn.functional.html#interpolate).
    NOTE: If `allow_scaling` is `False`, output features shapes of these two or more submodules must be the same, except for the channels/filters dimension if this is a dense link.
    NOTE: The only diference between residual and dense links ('is_residual' beeing 'True' of 'False') is the default '_reduction' function beeing respectively 'sum' and 'concat'.
    """
    def _create_link_submodule(submodule_params: Dict[str, Any], is_residual: bool, allow_scaling: bool = False, scaling_mode: str = 'linear', channel_dim: int = 1) -> MODULE_CREATOR_CALLBACK_RETURN_T:
        def _forward_callback(x: torch.Tensor, referenced_submodules_out: List[torch.Tensor], tensor_reduction_fn: REDUCTION_FUNCTION_T = (TENSOR_REDUCTION_FUNCTIONS['sum'] if is_residual else TENSOR_REDUCTION_FUNCTIONS['concat'])):

            # If target output shape (which is the same as `x` features shape) is different from one of the referenced tensor shapes, we perform a up/down-scaling (interpolation) if allowed
            tensors = [x, ]
            for y in referenced_submodules_out:
                if x.shape[channel_dim + 1:] != y.shape[channel_dim + 1:]:
                    if allow_scaling:
                        # Resize y features tensor to be of the same shape as x along dimensions after channel dim (scaling performed with `torch.nn.functional.interpolate`)
                        tensors.append(F.interpolate(y, size=x.shape[channel_dim + 1:], mode=scaling_mode))
                    else:
                        raise RuntimeError(f"Error: Couldn't forward throught {'residual' if is_residual else 'dense'} link: features from link doesn't have "
                                           f"the same shape as previous module's output shape, can't concatenate or add them. (did you forgot to allow residual/dense "
                                           f"features to be scaled using `allow_scaling: true` parameter?). `residual_shape='{y.shape}' != prev_features_shape='{x.shape}'`")
                else:
                    tensors.append(y)

            # Add or concatenate previous sub-module output features with residual or dense features
            rslt = tensor_reduction_fn(tensors, dim=channel_dim)
            if not isinstance(rslt, torch.Tensor):
                raise ValueError('Error: Wrong reduction value: "none" reduction is forbiden for residual/dense links. Use "mean", "sum" (default "_reduction" for residual links) or "concat" (default "_reduction" for dense links) reduction instead. '
                                 f'Result from reduction function in dense/residual link forward callback returned multiple tensors: got "{rslt}" from "{tensor_reduction_fn}" reduction function.')
        return _forward_callback

    _create_link_submodule.__doc__ = _residual_dense_link.__doc__
    return partial(_create_link_submodule, is_residual=is_residual)


BASIC_SUBMODULE_CREATORS = {'avg_pooling': _create_avg_pooling, 'conv2d': _create_nn_layer(is_fully_connected=False), 'fully_connected': _create_nn_layer(is_fully_connected=True),
                            'residual_link': _residual_dense_link(is_residual=True), 'dense_link': _residual_dense_link(is_residual=False)}


class DeepcvModule(torch.nn.Module):
    """ DeepCV PyTorch Module model base class
    Handles NN architecture definition tooling for easier model definition (e.g. from a YAML configuration file), model intialization and required/defaults hyperparameters logic.
    Child class can define `HP_DEFAULTS` class attribute to take additional parameters. By default, a `DeepcvModule` expects the following keys in `hp`: `architecture` and `act_fn`. `batch_norm` and `dropout_prob` can also be needed depending on which submodules are used.

    (Hyper)Parameters specified in a submodule's specs (from `hp['achitecture']`) and global parameters (directly from `hp`) which are not in `DeepcvModule.HP_DEFAULTS` can be provided to their respective submodule creator (or a torch.nn.Module type, created from its __init__ constructor).
    For example, if there is a module creator called `conv2d` and a submodule in DeepcvModule's `hp['achitecture']` list have the following YAML spec:
    ``` yaml
        - conv2d: { kernel_size: [3, 3], out_channels: 16, padding: 1, act_fn: !py!torch.nn.ReLU }
    ```
    Then, module creator function, which should return defined torch.nn.Module, can take its arguments in many possible ways:
        - it can take as arguments `submodule_params` (parameters dict from YAML submodule specs) and/or `prev_shapes` (previous model's sub-modules output shapes)
        - it can also (instead or additionally) directly take any arguments, for example, `act_fn: Optional[torch.nn.Module]`, so that `act_fn` can both be specified localy in submodule params (like in `submodule_params`) or globaly in DeepcvModule's specs. (directly in `hp`). This mechanism allows easier architecture specifications by specifying parameters for all submodules at once, while still being able to override global value localy (in submodule specs parameters dict) if needed.
    NOTE: In order to make usage of this mechanism, parameters/arguments names which can be specified globally should not be in `DeepcvModule.HP_DEFAULTS`, for example, a submodule creator can't take an argument named `architecture` because `architecture` is among `DeepcvModule.HP_DEFAULTS` entries so `hp['architecture']` won't be provided to any submodules creators nor nested DeepcvModule(s).
    NOTE: Submodule creator functions (and/or specified submodules `torch.nn.Module` types `__init__` constructors) must take named arguments to be supported (`*args` and `**kwargs` not supported)
    NOTE: If a submodule creators takes a parameter directly as an argument instead of taking it from `submodule_params` dict argument, then this parameter won't be present in `submodule_params` dict, even if its value have been specified localy in submodule parameters (i.e. even if its value doesn't comes from global `hp` entries).

    .. For more details about `architecture` hyperparameter parsing, see code in `DeepcvModule.define_nn_architecture` and examples of DeepcvModule(s) YAML architecture specification in ./conf/base/parameters.yml
    NOTE: A sub-module's name defaults to 'submodule_{i}' where 'i' is sub-module index in architecture sub-module list. Alternatively, you can specify a sub-module's name in architecture configuration, which, for example, allows you to define residual/dense links.
    .. See examples of Deepcv model sub-modules architecture definition in `[Kedro hyperparameters YAML config file]conf/base/parameters.yml`
    """

    HP_DEFAULTS = {'architecture': ...}

    def __init__(self, input_shape: torch.Size, hp: Union[deepcv.meta.hyperparams.Hyperparameters, Dict[str, Any]]):
        super().__init__()
        self._input_shape = input_shape

        # Process module hyperparameters
        assert self.HP_DEFAULTS != ..., f'Error: Module classes which inherits from "DeepcvModule" ({type(self).__name__}) must define "HP_DEFAULTS" class attribute dict.'
        self._hp, _missing = deepcv.meta.hyperparams.to_hyperparameters(hp, defaults=self.HP_DEFAULTS, raise_if_missing=True)

        # Create model architecture according to hyperparameters's `architecture` entry (see ./conf/base/parameters.yml for examples of architecture specs.) and initialize its parameters
        self.define_nn_architecture(self._hp['architecture'])
        self.initialize_parameters(self._hp['act_fn'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == len(self._input_shape):
            # Turn single input tensor into a batch of size 1
            x = x.unsqueeze(dim=0)

        # Apply object detector neural net architecture on top of shared image embedding features and input image
        if len(self._forward_callbacks) > 0:
            referenced_output_features = {}
            for name, subm in self._submodules.items():
                if name in self._forward_callbacks:
                    current_subm_references = []
                    if name in self._submodule_references:
                        # Find referer's referenced features from `self._submodule_references`
                        for referenced_submodule in self._submodule_references[name]:
                            current_subm_references.append(referenced_output_features[referenced_submodule])

                        # Free stored output features if there isn't any referrers referencing a submodule anymore (i.e. all forward callbacks have consumed stored output features for a given referenced sub-module)
                        refs = self._submodule_references[name]
                        del self._submodule_references[name]
                        for referenced_submodule in refs:
                            if any(map(self._submodule_references.values(), lambda refs: referenced_submodule in refs)):
                                # There isn't any referrer submodules to take stored features as input anymore, so we free memory for this reference
                                del referenced_output_features[referenced_submodule]

                    # Handle NNI InputChoices (chooses one or more tensor among referenced tensors)
                    if name in self._mutable_inputs:
                        current_subm_references = self._mutable_inputs[name](current_subm_references)

                    forward_callback_signature = inspect.signature(self._forward_callbacks[name]).parameters
                    optional_forward_callback_args = {'referenced_submodules_out': current_subm_references} if 'referenced_submodules_out' in forward_callback_signature else {}

                    # If needed, give reduction function ('sum', 'mean', 'concat' or 'none' reduction) to forward callback
                    if 'referenced_submodules_out' in forward_callback_signature and name in self._reduction_functions:
                        optional_forward_callback_args['tensor_reduction_fn'] = self._reduction_functions[name]

                    # Forward pass through sub-module
                    x = self._forward_callbacks[name](x, **optional_forward_callback_args)
                else:
                    # Forward pass through sub-module
                    x = subm(x)

                # If submodule is referenced by another module by a '_from' entry in its parameters, then we store its output features for later use (e.g. for a residual link)
                if name in sum(self._submodule_references.values(), tuple()):
                    referenced_output_features[name] = x
                return x
        else:
            return self._net(x)

    def __str__(self) -> str:
        """ Describes DeepCV module in a human readable text string, see `DeepcvModule.describe()` function or `DeepcvModuleDescriptor` class for more details """
        return str(self.describe())

    def describe(self):
        """ Describes DeepCV module with its architecture, capacity and features shapes at sub-modules level.
        Args:
            - to_string: Whether deepcv NN module should be described by a human-readable text string or a NamedTuple of various informations which, for example, makes easier to visualize model's sub-modules capacities or features shapes...
        Returns a `DeepcvModuleDescriptor` which contains model name, capacity, and eventually submodules names, feature shapes/dims/sizes and capacities...
        """
        return DeepcvModuleDescriptor(self)

    def define_nn_architecture(self, architecture_spec: Iterable, submodule_creators: Optional[Dict[str, Callable]] = None, extend_basic_submodule_creators_dict: bool = True):
        """ Defines neural network architecture by parsing 'architecture' hyperparameter and creating sub-modules accordingly
        .. For examples of DeepcvModules YAML architecture specification, see ./conf/base/parameters.yml
        NOTE: defines `self._features_shapes`, `self._submodules_capacities`, `self._forward_callbacks`, `self._submodules` and `self._architecture_spec` attributes (usefull for debuging and `self.__str__` and `self.describe` functions)
        Args:
            - architecture_spec: Neural net architecture definition listing submodules to be created with their respective parameters (probably from hyperparameters of `conf/base/parameters.yml` configuration file)
            - submodule_creators: Dict of possible architecture sub-modules associated with their respective module creators. If None, then defaults to `deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS`.
            - extend_basic_submodule_creators_dict: Boolean indicating whether `submodule_creators` argument will be extended with `deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS` dict or not. i.e. whether `submodule_creators` defines additionnal sub-modules or all existing sub-modules. (if `True` and some submodule name(s) (i.e. Dict key(s)) are both present in `submodule_creators` and  `deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS`, then `submodule_creators` dict values (submodule creator(s) Callable(s)) will override defaults/basic one(s)).
        """
        self._features_shapes = [self._input_shape]
        self._architecture_spec = architecture_spec
        self._submodules_capacities = list()
        self._forward_callbacks = dict()
        self._submodules = dict()
        self._submodule_references = dict()  # Dict which associates referenced sub-modules name/label with a set of their respective referrer sub-modules name/label
        self._reduction_functions = dict()  # Stores reduction functions for submodules creators which references tensors using '_from' (or '_from_nni_mutable_input') (only applicable to submodules based on forward callbacks, not 'torch.nn.Module')
        self._mutable_inputs = dict()  # Stores any NNI NAS Mutable InputChoice associated with its respective submodule name (only applicable to submodules based on forward callbacks with `_from_nni_mutable_input` parameter specified, not 'torch.nn.Module')

        if submodule_creators is None:
            submodule_creators = BASIC_SUBMODULE_CREATORS
        elif extend_basic_submodule_creators_dict:
            submodule_creators = {**BASIC_SUBMODULE_CREATORS, **submodule_creators}

        # Parse submodule NN architecture spec in order to define PyTorch model's submodules accordingly
        submodules = []
        for i, submodule_spec in enumerate(architecture_spec):
            submodule_type, params = list(submodule_spec.items())[0] if isinstance(submodule_spec, Dict) else (submodule_spec, {})
            submodule_name = f'_submodule_{i}'
            if isinstance(params, List) or isinstance(params, Tuple):
                # Architecture definition specifies a sub-module name explicitly
                submodule_name, params = params[0], params[1]
            elif isinstance(params, str):
                # Architecture definition specifies a sub-module name explicitly without any other sub-module parameters
                submodule_name, params = params, dict()

            # Checks if `params` is a valid and if there are eventual invalid or duplicate `submodule_name`(s)
            if not isinstance(params, Dict):
                raise RuntimeError(f'Error: Architecture sub-module spec. must either be a parameters Dict, or a submodule name along with parameters Dict, but got: "{params}".')
            if submodule_name is not None and submodule_name in OrderedDict(submodules).keys() or submodule_name == r'' or not isinstance(submodule_name, str):
                raise ValueError(f'Error: Invalid or duplicate sub-module name/label: "{submodule_name}"')

            # Add global (hyper)parameters from `hp` to `params` (allows to define parameters like `act_fn`, `dropout_prob`, `batch_norm`, ... either globaly in `hp` or localy in `params` from submodule specs)
            # NOTE: In case a parameter is both specified in `self._hp` globals and in `params` local submodule specs, `params` entries from submodule specs will allways override parameters from `hp`
            # NOTE: Merged parameters given to submodule (`params_with_globals`) wont contain any parameters specified in `DeepcvModule.HP_DEFAULTS` (e.g. submodule parameters won't contain parent architecture specs, i.e., no `DeepcvModule._hp['architecture']` in `params_with_globals`))
            params_with_globals = {n: copy.deepcopy(v) for n, v in self._hp.items() if n not in self.HP_DEFAULTS and n not in params}
            params_with_globals.update(params)

            # Create submodule (nested DeepcvModule or submodule from `submodule_creators` or `torch.nn.Module` submodule)
            # List of alternative submodules: nni_mutables.LayerChoice (+ reduction optional parameter ('sum' by default (other valid string values: 'mean', 'concat' or 'none'), + make sure candidate submodules names can't be referenced : LayerChoice candidates may have names (OrderedDict instead of List) but references are only allowed on '_nni_mutable_layer' global name)
            #  for more details on `LayerChoice`, see https://nni.readthedocs.io/en/latest/NAS/NasReference.html#nni.nas.pytorch.mutables.LayerChoice
            if submodule_type == '_nni_mutable_layer':
                pass
                # TODO: Implement this: must be implemented throught siamese branch implementation which involves factorization of this function code... :-(

            if submodule_type == '_deepcvmodule':
                # Allow nested DeepCV sub-module (see deepcv/conf/base/parameters.yml for examples)
                deepcv_submodule = type(self)(input_shape=self._features_shapes[-1], hp=params_with_globals)
                submodules.append((submodule_name, deepcv_submodule))
            else:
                if isinstance(submodule_type, str):
                    # Try to find sub-module creator or a torch.nn.Module's `__init__` function which matches `submodule_type` identifier
                    fn_or_type = submodule_creators.get(submodule_type)
                    if not fn_or_type:
                        # If we can't find suitable function in module_creators, we try to evaluate function name (allows external functions to be used to define model's modules)
                        try:
                            fn_or_type = deepcv.utils.get_by_identifier(submodule_type)
                        except Exception as e:
                            raise RuntimeError(f'Error: Could not locate module/function named "{submodule_type}" given module creators: "{submodule_creators.keys()}"') from e
                else:
                    fn_or_type = submodule_type

                # Create layer/block submodule from its module_creator or its torch.nn.Module.__init__ function (fn_or_type)
                submodule_signature_params = inspect.signature(fn_or_type).parameters
                provided_params = {n: v for n, v in params_with_globals.items() if n in submodule_signature_params}
                submdule_params = dict(**params)
                for n in provided_params.keys():
                    if n in submdule_params:
                        # Avoid to provide the same parameter twice (either provided through `submdule_params` dict or directly as an argument named after this parameter `n`)
                        del submdule_params[n]
                provided_params.update({n: v for n, v in {'submodule_params': submdule_params, 'prev_shapes': self._features_shapes}.items() if n in submodule_signature_params})
                module_or_callback = fn_or_type(**provided_params)

                # Figure out fn_or_type's output (module creators can return a torch.nn.Module or a callback which is called during forwarding of sub-modules (these callbacks are fed with a referenced sub-module output in addition to previous sub-module output)
                if isinstance(module_or_callback, torch.nn.Module):
                    submodules.append((submodule_name, module_or_callback))
                elif isinstance(module_or_callback, Callable) and len(inspect.Signature.from_callable(module_or_callback).parameters) == 2:
                    # Module_or_callback is assumed to be a callback which signature should match 'MODULE_CREATOR_CALLBACK_RETURN_T' (we only check if it's a Callable which takes two parameters)
                    self._forward_callbacks[submodule_name] = module_or_callback
                    submodules.append((submodule_name, None))

                    # '_from_nni_mutable_input' occurences in `params` are handled like '_from' entries: nni_mutables.InputChoice(references) + optional parameters 'n_chosen' (None by default, should be an integer between 1 and number of candidates) and 'reduction'
                    if '_from_nni_mutable_input' in params:
                        n_chosen = params['n_chosen'] if '_n_chosen' in params else None
                        n_candidates = len(params['_from_nni_mutable_input'])
                        self._mutable_inputs[submodule_name] = nni_mutables.InputChoice(n_candidates=n_candidates, n_chosen=n_chosen, key=submodule_name, reduction='none')

                    # Store any sub-module name/label references (used to store referenced submodule's output features during model's forward pass in order to reuse these features later in a forward callback (e.g. for residual links))
                    if '_from' in params:
                        # Allow multiple referenced sub-module(s) (`_from` entry can either be a list/tuple of referenced sub-modules name/label or a single sub-module name/label)
                        self._submodule_references[submodule_name] = tuple((params['_from'],)) if issubclass(type(params['_from']), str) else tuple(params['_from'])

                        if '_reduction' in params:
                            if params['_reduction'] not in TENSOR_REDUCTION_FUNCTIONS:
                                raise ValueError(f'Error: Tensor reduction function ("_reduction" parameter) should be one these string values: {TENSOR_REDUCTION_FUNCTIONS.keys()}, got `"_reduction"="{params["_reduction"]}"`. '
                                                 '"_reduction" is the reduction function applied to referenced tensors in "_from" (or chosen tensors in "_from_nni_mutable_input" if `"n_chosen" > 1`)')
                            self._reduction_functions[submodule_name] = TENSOR_REDUCTION_FUNCTIONS[params['_reduction']]
                else:
                    raise RuntimeError(f'Error: Wrong sub-module creator function/class __init__ return type '
                                       f'(must either be a torch.nn.Module or a `forward` callback of type: `{MODULE_CREATOR_CALLBACK_RETURN_T}`.')

            # Store newly added NN submodule's capacity and output features shape information
            self._submodules_capacities.append(deepcv.meta.nn.get_model_capacity(submodules[-1][1]))
            self._net = torch.nn.Sequential(OrderedDict([(n, m) for (n, m) in submodules if m is not None]))
            self._features_shapes.append(deepcv.meta.nn.get_out_features_shape(self._input_shape, self._net))

        self._submodules = OrderedDict(submodules)

        # Make sure all referenced sub-module exists (i.e. that there is a matching submodule name/label)
        for references in self._submodule_references.values():
            missing = [ref for ref in references if ref not in self._submodules.keys()]
            if len(missing) > 0:
                raise ValueError(f"Error: Invalid sub-module reference(s), can't find following sub-module name(s)/label(s): {missing}")

    def initialize_parameters(self, act_fn: Optional[Type[torch.nn.Module]] = None):
        """ Initializes model's parameters with Xavier Initialization with a scale depending on given activation function (only needed if there are convolutional and/or fully connected layers). """
        xavier_gain = torch.nn.init.calculate_gain(deepcv.meta.nn.get_gain_name(act_fn)) if act_fn else None

        def _raise_if_no_act_fn(sub_module_name: str):
            if xavier_gain is None:
                msg = f'Error: Must specify  in `DeepcvModule.initialize_parameters` function in order to initialize {sub_module_name} layer sub-module with xavier initialization.'
                raise RuntimeError(msg)

        def _xavier_init(module: torch.nn.Module):
            if deepcv.meta.nn.is_conv(module):
                _raise_if_no_act_fn('convolution')
                torch.nn.init.xavier_normal_(module.weight.data, gain=xavier_gain)
                module.bias.data.fill_(0.)
            elif deepcv.meta.nn.is_fully_connected(module):
                _raise_if_no_act_fn('fully connected')
                torch.nn.init.xavier_uniform_(module.weight.data, gain=xavier_gain)
                module.bias.data.fill_(0.)
            elif type(module).__module__ == torch.nn.BatchNorm2d.__module__:
                torch.nn.init.uniform_(module.weight.data)  # gamma == weight here
                module.bias.data.fill_(0.)  # beta == bias here
            elif list(module.parameters(recurse=False)) and list(module.children()):
                raise Exception("ERROR: Some module(s) which have parameter(s) haven't bee explicitly initialized.")
        self.apply(_xavier_init)

    @ property
    def needed_python_sources(self, project_path: Union[str, Path]) -> Set[str]:
        """ Returns Python source files needed for model inference/deployement/serving.
        This function can be usefull, for example, if you want to log model to mlflow and be able to deploy it easyly with any supported way: Local mlflow REST API enpoint, Docker image, Azure ML, Amazon Sagemaker, Apache Spark UDF, ...
        .. See [mlflow.pytorch API](https://www.mlflow.org/docs/latest/python_api/mlflow.pytorch.html) and [MLFLow Model built-in-deployment-tools](https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools)
        NOTE: Depending on your need, there are other ways to deploy a model or a pipeline from DeepCV: For example, Kedro and PyTorch also provides tooling for machine learning model(s)/pipeline(s) deployement, serving, portability and reproductibility.
        NOTE: PARTIALY TESTED: Be warned this function tries to retreive all module's source file dependencies within this project directory recursively but may fail to find some sources in some corner cases; So you may have to add some source files by your own.
        # TODO: better test this function
        """
        python_sources = set()

        def _add_if_source_in_project_path(source):
            try:
                if source not in python_sources and Path(source).relative_to(project_path):
                    python_sources.add(source)
                    return True
            except ValueError:
                pass
            return False

        # For each sub component (sub torch.nn.module) of this model, we look for its source file and all source files it depends on recursively (only sources within project directory are taken into account)
        for subm in self._submodules.values():
            module = inspect.getmodule(type(subm))
            source = inspect.getsourcefile(type(subm))
            if source is not None:
                _add_if_source_in_project_path(source)
            if module is not None:
                # Recursively search for all module dependencies which are located in project directory
                modules = {module, }
                while len(modules) > 0:
                    for m in modules:
                        for name in dir(m).items():
                            sub_module = getattr(m, name, None)
                            if isinstance(sub_module, types.ModuleType) and hasattr(sub_module, '__file__'):  # if sub module doesn't have __file__ it is assumed to be built-in (ignored)
                                if _add_if_source_in_project_path(sub_module.__file__) and m not in modules:
                                    modules.add(sub_module)
                        modules.remove(m)
        return python_sources


class DeepcvModuleWithSharedImageBlock(DeepcvModule):
    """ Deepcv Module With Shared Image Block model base class
    Appends to DeepcvModule a basic shared convolution block allowing transfert learning between all DeepCV models on images.
    """

    SHARED_BLOCK_DISABLED_WARNING_MSG = r'Warning: `DeepcvModule.{}` called while `self._enable_shared_image_embedding_block` is `False` (Shared image embedding block disabled for this model)'

    def __init__(self, input_shape: torch.Size, hp: deepcv.meta.hyperparams.Hyperparameters, enable_shared_block: bool = True, freeze_shared_block: bool = True):
        super().__init__(input_shape, hp)

        self._shared_block_forked = False
        self._enable_shared_image_embedding_block = enable_shared_block
        self.freeze_shared_image_embedding_block = freeze_shared_block

        if enable_shared_block and not hasattr(self, 'shared_image_embedding_block'):
            # If class haven't been instanciated yet, define common/shared DeepcvModule image embedding block
            type(self)._define_shared_image_embedding_block()

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if self._enable_shared_image_embedding_block:
            # Apply shared image embedding block and combine it's output with input image (concats features over channel dimension)
            x = torch.cat([x, self.shared_image_embedding_block(x)], dim=channel_dim)
        return super().forward(x)

    @ property
    def freeze_shared_image_embedding_block(self) -> bool:
        return self._freeze_shared_image_embedding_block

    @ freeze_shared_image_embedding_block.setter
    def set_freeze_shared_image_embedding_block(self, freeze_weights: bool):
        if self._enable_shared_image_embedding_block:
            self._freeze_shared_image_embedding_block = freeze_weights
            for p in self.shared_image_embedding_block.parameters():
                p.requires_grad = False
            # TODO: freeze/unfreeze weights...
            # TODO: handle concurency between different models training at the same time with unfreezed shared weights
        else:
            logging.warn(type(self).SHARED_BLOCK_DISABLED_WARNING_MSG.format('set_freeze_shared_image_embedding_block'))

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
            logging.warn(type(self).SHARED_BLOCK_DISABLED_WARNING_MSG.format('fork_shared_image_embedding_block'))
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
            logging.warn(type(self).SHARED_BLOCK_DISABLED_WARNING_MSG.format('merge_shared_image_embedding_block'))
        return False

    @ classmethod
    def _define_shared_image_embedding_block(cls, in_channels: int = 3):
        logging.info('Creating shared image embedding block of DeepcvModule models...')
        conv_opts = {'act_fn': torch.nn.ReLU, 'batch_norm': {'affine': True, 'eps': 1e-05, 'momentum': 0.0736}}
        layers = [('shared_block_conv_1', deepcv.meta.nn.conv_layer(conv2d={'in_channels': in_channels, 'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1}, **conv_opts)),
                  ('shared_block_conv_2', deepcv.meta.nn.conv_layer(conv2d={'in_channels': 8, 'out_channels': 16, 'kernel_size': (3, 3), 'padding': 1}, **conv_opts)),
                  ('shared_block_conv_3', deepcv.meta.nn.conv_layer(conv2d={'in_channels': 16, 'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1}, **conv_opts)),
                  ('shared_block_conv_4', deepcv.meta.nn.conv_layer(conv2d={'in_channels': 8, 'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1}, **conv_opts))]
        cls.shared_image_embedding_block = torch.nn.Sequential(OrderedDict(*layers))


class DeepcvModuleDescriptor:
    """ Describes DeepCV module with its architecture, capacity and features shapes at sub-modules level """

    def __init__(self, module: DeepcvModule):
        self.module = module

        if hasattr(module, '_architecture_spec'):
            # NOTE: `module.architecture_spec` attribute will be defined if `module.define_nn_architecture` is called
            architecture_spec = module._architecture_spec
        elif 'architecture' in module._hp:
            # otherwise, we try to look for architecture/sub-modules configuration in hyperparameters dict
            architecture_spec = module._hp['architecture']
        else:
            architecture_spec = None
            logging.warn(f"Warning: `{type(self).__name__}({type(module)})`: cant find NN architecture, no `module.architecture_spec` attr. nor `architecture` in `module._hp`")

        # Fills and return a DeepCV module descriptor
        self.capacity = deepcv.meta.nn.get_model_capacity(module)
        self.human_readable_capacity = deepcv.utils.human_readable_size(self.capacity)
        self.model_class = module.__class__
        self.model_class_name = module.__class__.__name__
        if isinstance(module, DeepcvModuleWithSharedImageBlock):
            self.uses_shared_block = module._enable_shared_image_embedding_block
            self.did_forked_shared_block = module._shared_block_forked
            self.freezed_shared_block = module._freeze_shared_image_embedding_block
            assert not self.did_forked_shared_block or self.uses_shared_block, 'Error: DeepCVModule have inconsistent flags: `_shared_block_forked` cant be True if `_enable_shared_image_embedding_block` is False'

        if hasattr(module, '_features_shapes'):
            self.submodules_features_shapes = module._features_shapes
            self.submodules_features_dims = map(len, module._features_shapes)
            self.submodules_features_sizes = map(np.prod, module._features_shapes)
        if architecture_spec is not None:
            self.architecture = architecture_spec
            self.submodules = {n: str(m) for n, m in module._submodules.items()}
        if hasattr(module, '_submodules_capacities'):
            self.submodules_capacities = module._submodules_capacities
            self.human_readable_capacities = map(deepcv.utils.human_readable_size, module._submodules_capacities)

    def __str__(self) -> str:
        """ Ouput a human-readable string representation of the deepcv module based on its descriptor """
        if self.architecture is not None:
            features = self.submodules_features_shapes if hasattr(self, 'submodules_features_shapes') else ['UNKNOWN'] * len(self.architecture)
            capas = self.human_readable_capacities if hasattr(self, 'human_readable_capacities') else ['UNKNOWN'] * len(self.architecture)
            desc_str = '\n\t'.join([f'- {n}({p}) output_features_shape={s}, capacity={c}' for (n, p), s, c in zip(self.submodules.items(), features, capas)])
        else:
            desc_str = '(No submodule architecture informations to describe)'

        if isinstance(self.module, DeepcvModuleWithSharedImageBlock):
            desc_str += '\n SIEB (Shared Image Embedding Block) usage:'
            if self.uses_shared_block:
                desc_str += 'This module makes use of shared image embedding block applied to input image:'
                if self.did_forked_shared_block:
                    desc_str += "\n\t- FORKED=True: Shared image embedding block parameters have been forked, SGD updates from other models wont impact this model's weights and SGD updates of this model wont change shared weights of other models until the are eventually merged"
                else:
                    desc_str += '\n\t- FORKED=False: Shared image embedding block parameters are still shared, any non-forked/non-freezed DeepcvModule SGD uptates will have an impact on these parameters.'
                if self.freezed_shared_block:
                    desc_str += '\n\t- SHARED=True: Shared image embedding block parameters have been freezed and wont be taken in account in gradient descent training of this module.'
                else:
                    desc_str += '\n\t- SHARED=False: Shared image embedding block parameters are not freezed and will be learned/fine-tuned during gradient descent training of this model.'
            else:
                desc_str = ' This module doesnt use shared image embedding block.'
        return f'{self.model_class_name} (capacity={self.human_readable_capacity}):\n\t{desc_str}'


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
