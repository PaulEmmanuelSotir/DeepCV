#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" DeepCV model base class meta module - base_module.py - `DeepCV`__
Defines DeepCV model base class
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
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


__all__ = ['TENSOR_OR_LIST_OF_TENSORS_T', 'MODULE_CREATOR_FORWARD_CALLBACK_T', 'REDUCTION_FUNCTION_T', 'TENSOR_REDUCTION_FUNCTIONS', 'BASIC_SUBMODULE_CREATORS',
           'ForwardCallbackSubmodule', 'DeepcvModule', 'DeepcvModuleWithSharedImageBlock', 'DeepcvModuleDescriptor']
__author__ = 'Paul-Emmanuel Sotir'

""" Helper constants for type checks and annotations in DeepcvModule """
TENSOR_OR_LIST_OF_TENSORS_T = Union[torch.Tensor, List[torch.Tensor]]
MODULE_CREATOR_FORWARD_CALLBACK_T = Callable[[TENSOR_OR_LIST_OF_TENSORS_T, Dict[str, torch.Tensor]], TENSOR_OR_LIST_OF_TENSORS_T]
REDUCTION_FUNCTION_T = Callable[[List[torch.Tensor, 'dim'], TENSOR_OR_LIST_OF_TENSORS_T]]

""" Reduction functions. These are available throught `DeepcvModule.REDUCTION_FUNCTION_TOKEN` parameter in YAML NN architecture specification of submodules which are instances of `ForwardCallbackSubmodule` (e.g. residual/dense links, ...: submodules defined by a callback called at forward passes) """
TENSOR_REDUCTION_FUNCTIONS = {'mean': torch.mean, 'sum': torch.sum, 'concat': torch.cat, 'none': lambda l, dim: l}


""" Default submodule types (defined by a name associated to a submodule creator function) available in 'deepcv.meta.base_module.DeepcvModule' YAML NN architecture specification.
This list can be extended or overriden according to your needs by provided your own submodule creator functions to `DeepcvModule`'s `__init__()` method.
NOTE: By default, there are other possible submodules which are builtin DeepcvModule: see `DeepcvModule.NESTED_DEEPCV_MODULE_TOKEN`, `DeepcvModule.NNI_MUTABLE_LAYER_CHOICE_TOKEN` and `DeepcvModule.NEW_BRANCH_FROM_TENSOR_TOKEN`
"""
BASIC_SUBMODULE_CREATORS = {'avg_pooling': _create_avg_pooling, 'conv2d': _create_nn_layer(is_fully_connected=False), 'fully_connected': _create_nn_layer(is_fully_connected=True),
                            'residual_link': _residual_dense_link(is_residual=True), 'dense_link': _residual_dense_link(is_residual=False)}

#_______________________________________________ DEEPCVMODULE CLASSES _________________________________________________#


class ForwardCallbackSubmodule(torch.nn.Module):
    """ `DeepcvModule`-specific Pytorch module which is defined from a callback called on forward passes.
    This Pytorch module behavior is only defined from given callback which makes it more suitable for residual/dense links, for example.
    `ForwardCallbackSubmodule`s are handled in a specific way by `DeepcvModule` for builtin support for output tensor references (e.g. residual links with `_from` parameter, see `_residual_dense_link` for example usage in a submodule creator)
    , reduction functions support (see TENSOR_REDUCTION_FUNCTIONS for more details) and NNI NAS Mutable InputChoice support. It means that `DeepcvModule` will parse reduction function and output tensor references and provide those in `self.reduction_fn`, `self.mutable_input_choice` before any forward passes and give those along with `referenced_submodules_out` argument to forward pass callback.
    NOTE: `tensor_reduction_fn` and `referenced_submodules_out` arguments are not mandatory in forward callback signature from a submodule creator, but can be taken according to your needs (if reduction function, tensor references and/or NNI NAS Mutable InputChoice support is needed for this NN submodule).
    """

    def __init__(self, forward_callback: MODULE_CREATOR_FORWARD_CALLBACK_T):
        self.forward_callback = forward_callback
        forward_callback_signature = inspect.signature(self._forward_callback).parameters
        self.takes_tensor_references = 'referenced_submodules_out' in forward_callback_signature
        self.takes_reduction_fn = 'tensor_reduction_fn' in forward_callback_signature
        # `self.mutable_input_choice` is filled at runtime by `DeepcvModule` when parsing architecture from NN YAML spec. (i.e. wont be `None` during forward passes if an NNI NAS input choice is specified in NN specs.)
        self.mutable_input_choice = None
        # `self.reduction_fn` is filled at runtime by `DeepcvModule` when parsing architecture from NN YAML spec. (i.e. wont be `None` during forward passes if a reduction function is specified explicitly in NN specs.)
        self.reduction_fn = None
        # `self.referenced_submodules` is filled at runtime by `DeepcvModule` when parsing architecture from NN YAML spec. (i.e. wont be `None` during forward passes if tensor reference(s) are specified in NN specs., e.g. using `DeepcvModule.FROM_TOKEN` or 'DeepcvModule.FROM_NNI_MUTABLE_INPUT_TOKEN`)
        self.referenced_submodules = None

    def forward(self, x: TENSOR_OR_LIST_OF_TENSORS_T, referenced_output_features: Optional[Dict[str, TENSOR_OR_LIST_OF_TENSORS_T]] = None) -> TENSOR_OR_LIST_OF_TENSORS_T:
        # If needed, give referenced output tensor(s) and a reduction function ('sum', 'mean', 'concat' or 'none' reduction) arguments to forward callback
        forward_kwargs = {}
        if referenced_output_features is not None and self.referenced_submodules is not None and self.takes_tensor_references:
            referenced_output_features = [v for n, v in referenced_output_features.items() if n in self.referenced_submodules]
            forward_kwargs['referenced_submodules_out'] = referenced_output_features if self.mutable_input_choice is None else self.mutable_input_choice(referenced_output_features)
        if self.tensor_reduction_fn is not None and self.takes_reduction_fn:
            forward_kwargs['tensor_reduction_fn'] = self.tensor_reduction_fn

        # Forward pass through sub-module based on forward pass callback
        return self.forward_callback(x, **forward_kwargs)


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
    NOTE: A sub-module's name defaults to 'submodule_{i}' where 'i' is sub-module index in architecture sub-module list. Alternatively, you can specify a sub-module's name in architecture configuration (either using a tuple/list of submodule name and params or by specifying a `_name` argument in submodule params), which, for example, allows you to define residual/dense links referencing these tensor(s) by their name(s).
    .. See complete examples of `DeepcvModule` model NN architecture specification in `[Kedro hyperparameters YAML config file]conf/base/parameters.yml`
    """

    HP_DEFAULTS = {'architecture': ...}
    DEFAULT_LAYER_CHOICE_REDUCTION = r'sum'

    # Special tokens which can be used in YAML architecture specification of `deepcv.meta.base_module.DeepcvModule`s (builtin tokens of `DeepcvModule` implementation without submodule creators)
    FROM_TOKEN = r'_from'
    SUBMODULE_NAME_TOKEN = r'_name'
    REDUCTION_FUNCTION_TOKEN = r'_reduction'
    NESTED_DEEPCV_MODULE_TOKEN = r'_nested_deepcv_module'
    NEW_BRANCH_FROM_TENSOR_TOKEN = '_new_branch_from_tensor'
    FROM_NNI_MUTABLE_INPUT_TOKEN = r'_from_nni_mutable_input'
    FROM_NNI_INPUT_N_CHOSEN_TOKEN = r'_n_chosen'
    NNI_MUTABLE_LAYER_CHOICE_TOKEN = '_nni_mutable_layer'
    NNI_LAYER_CHOICE_CANDIDATES_TOKEN = r'_candidates'
    NNI_NAS_MUTABLE_RETURN_MASK_TOKEN = r'_return_mask'

    def __init__(self, input_shape: torch.Size, hp: Union[deepcv.meta.hyperparams.Hyperparameters, Dict[str, Any]]):
        super().__init__()
        self._input_shape = input_shape
        self._uses_nni_nas_mutables = False
        self._uses_forward_callback_submodules = False

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

        # Apply DeepcvModule neural net architecture defined from a parsed spec. in `define_nn_architecture`
        if not self.is_sequencial_nn:
            # DeepcvModule NN architecure cant be applied with a single `torch.nn.Sequential`: secific handling is needed during forwarding due to output tensor reference(s), `DeepcvModule.NEW_BRANCH_FROM_TENSOR_TOKEN` submodule usage, and/or builtin reduction function support.
            referenced_output_features = {}
            # Stores remaining submodules output tensor references (shallow copy to avoid removing entries from `self._submodule_references` when reference(s) are consumed during forward pass/propagation)
            remaining_submodule_references = copy.copy(self._submodule_references[name])

            # Loop over all NN architecture submodules and handle output tensor references if needed (e.g. residual/dense links, ...)
            for name, subm in self._submodules.items():
                current_subm_references = OrderedDict()
                if getattr(subm, 'referenced_submodules', None) is not None and len(subm.referenced_submodules) > 0:
                    # Find referer's referenced features from `subm.referenced_submodules`
                    current_subm_references = OrderedDict([(ref, referenced_output_features[ref]) for ref in subm.referenced_submodules])

                    # Free stored output features if there isn't any referrers referencing a submodule anymore (i.e. all forward callbacks have consumed stored output features for a given referenced sub-module)
                    del remaining_submodule_references[name]
                    for referenced_submodule in subm.referenced_submodules:
                        if not any(map(remaining_submodule_references.values(), lambda refs: referenced_submodule in refs)):
                            # There isn't any referrer submodules to take stored features as input anymore, so we free memory for this referenced tensor
                            del referenced_output_features[referenced_submodule]

                if isinstance(subm, nni_mutables.LayerChoice) or isinstance(subm, ForwardCallbackSubmodule):
                    # Forward pass through a `ForwardCallbackSubmodule` or NNI NAS LayerChoice submodule (forward pass throught one or more underlying candidate submodules)
                    x = subm(x, referenced_output_features=current_subm_references)
                else:
                    # Forward pass through regular `torch.nn.Module` submodule
                    x = subm(x)

                # If submodule is referenced by another module by a `DeepcvModule.FROM_TOKEN` entry in its parameters, then we store its output features for later use (e.g. for a residual/dense link)
                if name in sum(remaining_submodule_references.values(), tuple()):
                    referenced_output_features[name] = x
                return x
        elif hasattr(self, '_sequential_net'):
            # NN architecture is only a `torch.nn.Sequential` model (no submodules which makes usage of tensor references, reduction functions nor NNI NAS Mutables, i.e., no `_from` nor `DeepcvModule.FROM_NNI_MUTABLE_INPUT_TOKEN` usage, ect...)
            return self._sequential_net(x)
        else:
            raise RuntimeError('Error: Incoherent `DeepcvModule` model; No `_sequential_net` attribute while NN architecture seams to be sequential'
                               '(no residual/dense link usage, nor NNI NAS MutableInput usage, nor builtin reduction function specified, ...). There may be a bug in model definition from NN architecture specification in `DeepcvModule.define_nn_architecture`')

    def __str__(self) -> str:
        """ Describes DeepCV module in a human readable text string, see `DeepcvModule.describe()` function or `DeepcvModuleDescriptor` class for more details """
        return str(self.describe())

    @property
    def uses_nni_nas_mutables(self) -> bool:
        """ Readonly boolean property which indicates whether `DeepcvModule` instance makes usage of NNI NAS Mutable API (`LayerChoice`(s) or `InputChoice`(s) throuht usgae of `NNI_MUTABLE_LAYER_CHOICE_TOKEN` or `FROM_NNI_MUTABLE_INPUT_TOKEN` submodule of YAML architecture spec.) """
        return self._uses_nni_nas_mutables

    @property
    def uses_forward_callback_submodules(self) -> bool:
        """ Readonly boolean property which indicates whether `DeepcvModule` instance makes usage of NNI NAS Mutable API (`LayerChoice`(s) or `InputChoice`(s) throuht usgae of `NNI_MUTABLE_LAYER_CHOICE_TOKEN` or `FROM_NNI_MUTABLE_INPUT_TOKEN` submodule of YAML architecture spec.) """
        return self._uses_forward_callback_submodules

    @property
    def is_sequencial_nn(self) -> bool:
        """ Readonly boolean property indicating whether `DeepcvModule` NN architecture (parsed from YAML NN specs.) can be applied as a `torch.nn.Sequential` (`self._sequential_net` attribute then exists and is a `torch.nn.Sequential`).
        I.e., returns whether if `DeepcvModule` architecture doesn't uses any residual/dense links (Or any other `ForwardCallbackSubmodule`), nor `DeepcvModule.NEW_BRANCH_FROM_TENSOR_TOKEN` submodules, nor NNI NAS mutable inputs/layers, nor builtin reduction function support which all needs specific handling during forward passes (impossible to apply whole architecture by using a single `torch.nn.Sequential`). """
        return not self.uses_nni_nas_mutables and not self._uses_forward_callback_submodules

    def describe(self):
        """ Describes DeepCV module with its architecture, capacity and features shapes at sub-modules level.
        Args:
            - to_string: Whether deepcv NN module should be described by a human-readable text string or a NamedTuple of various informations which, for example, makes easier to visualize model's sub-modules capacities or features shapes...
        Returns a `DeepcvModuleDescriptor` which contains model name, capacity, and eventually submodules names, feature shapes/dims/sizes and capacities...
        """
        return DeepcvModuleDescriptor(self)

    def define_nn_architecture(self, architecture_spec: Iterable, submodule_creators: Optional[Dict[str, Callable[..., torch.nn.Module]]] = None, extend_basic_submodule_creators_dict: bool = True):
        """ Defines neural network architecture by parsing 'architecture' hyperparameter and creating sub-modules accordingly
        .. For examples of DeepcvModules YAML architecture specification, see ./conf/base/parameters.yml
        NOTE: defines `self._features_shapes`, `self._submodules_capacities`, `self._submodules` and `self._architecture_spec` attributes (usefull for debuging and `self.__str__` and `self.describe` functions)
        Args:
            - architecture_spec: Neural net architecture definition listing submodules to be created with their respective parameters (probably from hyperparameters of `conf/base/parameters.yml` configuration file)
            - submodule_creators: Dict of possible architecture sub-modules associated with their respective module creators. If None, then defaults to `deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS`.
            - extend_basic_submodule_creators_dict: Boolean indicating whether `submodule_creators` argument will be extended with `deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS` dict or not. i.e. whether `submodule_creators` defines additionnal sub-modules or all existing sub-modules. (if `True` and some submodule name(s) (i.e. Dict key(s)) are both present in `submodule_creators` and  `deepcv.meta.base_module.BASIC_SUBMODULE_CREATORS`, then `submodule_creators` dict values (submodule creator(s) Callable(s)) will override defaults/basic one(s)).
        """
        self._features_shapes = [self._input_shape]
        self._architecture_spec = architecture_spec
        self._submodules_capacities = list()
        self._submodules = OrderedDict()
        self._submodule_references = dict()  # Dict which associates referenced sub-modules name/label with a set of their respective referrer sub-modules name/label (referenced tensor(s) using DeepcvModule.FROM_TOKEN or referenced tensors candidate(s)) using DeepcvModule.FROM_NNI_MUTABLE_INPUT_TOKEN)

        if submodule_creators is None:
            submodule_creators = BASIC_SUBMODULE_CREATORS
        elif extend_basic_submodule_creators_dict:
            submodule_creators = {**BASIC_SUBMODULE_CREATORS, **submodule_creators}

        # Parse submodule NN architecture spec in order to define PyTorch model's submodules accordingly
        for i, submodule_spec in enumerate(architecture_spec):
            # Parse submodule specification to obtain a new `torch.nn.Module` submodule of NN architecture
            new_submodule_name, new_submodule = self._parse_torch_module_from_submodule_spec(submodule_spec, i, submodule_creators)
            # Append new submodule to NN architecture (`self_submodules` `OrderedDict`)
            self._submodules[new_submodule_name] = new_submodule
            # Store tensor references for easier/better memory handling during forward passes (e.g. residual/dense links, `DeepcvModule.NEW_BRANCH_FROM_TENSOR_TOKEN` usage, ...)
            if (isinstance(subm, ForwardCallbackSubmodule) or isinstance(subm, nni_mutables.LayerChoice)) and getattr(subm, 'referenced_submodules') is not None:
                self._submodule_references[submodule_name] = subm.referenced_submodules
            # Figure out new NN submodule capacity
            # TODO: Modify `deepcv.meta.nn.get_model_capacity` in case submodule is a `MutableLayer`s (Return a list of capacities or mean capacity? For now, returns sum of capacities...)
            self._submodules_capacities.append(deepcv.meta.nn.get_model_capacity(module))
            if self.is_sequencial_nn:
                # self._sequential_net is only accurate/applicable/defined when there isn't any submodules using tensor reference(s), NNI NAS Mutable InputChoice(s), nor non-default reduction functions, but will still be accurate on features shapes (specific logic is needed in forward passes of `DeepcvModule`)
                self._sequential_net = torch.nn.Sequential(self._submodules)
            else:
                # `torch.nn.Sequential` usage is impossible due to some DeepcvModule features/submodules used in architecture spec. (residual/dense link(s), builtin reduction function usage, NNI NAS Mutable InputChoice usage, ...)
                del self._sequential_net
            # Make sure all referenced sub-module exists (i.e. that there is a matching submodule name/label)
            missing_refs = [ref for ref in self._submodule_references[submodule_name] if ref not in self._submodules.keys()]
            if len(missing_refs) > 0:
                raise ValueError(f'Error: Invalid sub-module reference(s), cant find following sub-module name(s)/label(s): "{missing_refs}".'
                                 ' Output tensor references must refer to a previously defined sub-module name.')
            # Figure out output features shape from new submodule by performing a dummy forward pass of `DeepcvModule` instance
            self._features_shapes.append(deepcv.meta.nn.get_out_features_shape(self._input_shape, self))

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

    def get_needed_python_sources(self, project_path: Union[str, Path]) -> Set[str]:
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

    nn_spec_parser_context = SimpleNamespace(submodule_creators=submodule_creators, tokens=SimpleNamespace(NESTED_DEEPCV_MODULE, ...), mutable_inputs, mutable_layers, _submodule_references, reduction_functions, hp, hp_defaults)

    def _parse_torch_module_from_submodule_spec(self, submodule_spec: Union[Dict, Type[torch.nn.Module], str, Callable[..., torch.nn.Module]], submodule_pos: Union[int, str], submodule_creators: Dict[str, Callable[..., torch.nn.Module]], default_submodule_prefix: str = '_submodule_', allow_mutable_layer_choices: bool = True) -> Tuple[str, torch.nn.Module]:
        """ Defines a single submodule of `DeepcvModule` from its respective NN architecture spec. """
        submodule_name = default_submodule_prefix + str(submodule_pos)
        params, submodule_type = self._submodule_name_and_params_from_spec(submodule_spec, submodule_name=submodule_name, existing_submodule_names=self._submodules.keys())

        # Add global (hyper)parameters from `hp` to `params` (allows to define parameters like `act_fn`, `dropout_prob`, `batch_norm`, ... either globaly in `hp` or localy in `params` from submodule specs)
        # NOTE: In case a parameter is both specified in `self._hp` globals and in `params` local submodule specs, `params` entries from submodule specs will allways override parameters from `hp`
        # NOTE: Merged parameters given to submodule (`params_with_globals`) wont contain any parameters specified in `DeepcvModule.HP_DEFAULTS` (e.g. submodule parameters won't contain parent architecture specs, i.e., no `DeepcvModule._hp['architecture']` in `params_with_globals`))
        params_with_globals = {n: copy.deepcopy(v) for n, v in self._hp.items() if n not in self.HP_DEFAULTS and n not in params}
        params_with_globals.update(params)

        if submodule_type == self.NESTED_DEEPCV_MODULE:
            # Allow nested DeepCV sub-module (see deepcv/conf/base/parameters.yml for examples)
            module = type(self)(input_shape=self._features_shapes[-1], hp=params_with_globals)
        elif submodule_type == self.NNI_MUTABLE_LAYER_CHOICE_TOKEN:
            self._uses_nni_nas_mutables = True
            if not allow_mutable_layer_choices:
                raise ValueError(f'Error: nested LayerChoices are forbiden, cant specify a NNI NAS Mutable LayerChoice as a candidate of another LayerChoice ("{submodule_type}").')
            # List of alternative submodules: nni_mutables.LayerChoice (+ reduction optional parameter ('sum' by default (other valid string values: 'mean', 'concat' or 'none'), + makes sure candidate submodules names can't be referenced : LayerChoice candidates may have names (OrderedDict instead of List) but references are only allowed on '_nni_mutable_layer' global name)
            # for more details on `LayerChoice`, see https://nni.readthedocs.io/en/latest/NAS/NasReference.html#nni.nas.pytorch.mutables.LayerChoice
            if not isinstance(params, Dict[str, Any]) or self.NNI_LAYER_CHOICE_CANDIDATES_TOKEN not in params or any([p not in {self.NNI_LAYER_CHOICE_CANDIDATES_TOKEN, self.REDUCTION_FUNCTION_TOKEN, self.NNI_NAS_MUTABLE_RETURN_MASK_TOKEN} for p in params.keys()]):
                raise ValueError(f'Error: Parameters of a "{self.NNI_MUTABLE_LAYER_CHOICE_TOKEN}" submodule specification must be a Dict which at least contains a `_candidates` parameter. '
                                 f'(And may eventually specify `{self.REDUCTION_FUNCTION_TOKEN}`, `{self.NNI_NAS_MUTABLE_RETURN_MASK_TOKEN}` and/or `{self.SUBMODULE_NAME_TOKEN}` parameter(s)). NNI Mutable LayerChoice submodule params received: "{params}"')
            prefix = f'{default_submodule_prefix}{submodule_pos}_candidate_'
            reduction = params[self.REDUCTION_FUNCTION_TOKEN] if self.REDUCTION_FUNCTION_TOKEN in params else self.DEFAULT_LAYER_CHOICE_REDUCTION
            return_mask = params[self.NNI_NAS_MUTABLE_RETURN_MASK_TOKEN] if self.NNI_NAS_MUTABLE_RETURN_MASK_TOKEN in params else False

            # Parse candidates/alternative submodules from params (recursive call to `self._parse_torch_module_from_submodule_spec`)
            candidate_refs, candidates = tuple(), OrderedDict()
            for j, candidate_spec in enumerate(params[self.NNI_LAYER_CHOICE_CANDIDATES_TOKEN]):
                candidate_name, candidate = _parse_torch_module_from_submodule_spec(submodule_spec=candidate_spec, submodule_pos=j, submodule_creators=submodule_creators,
                                                                                    default_submodule_prefix=prefix, allow_mutable_layer_choices=False)
                if isinstance(candidate, ForwardCallbackSubmodule) and getattr(candidate, 'referenced_submodules') is not None:
                    candidate_refs += candidate.referenced_submodules
                elif not isinstance(candidate, ForwardCallbackSubmodule):
                    # Ignores any tensor references if candidate isn't a `ForwardCallbackSubmodule`
                    # NOTE: Assumes `referenced_output_features` is a reserved argument in forward passes of submodules (reserved for usage in `ForwardCallbackSubmodule`s)
                    def _forward_monkey_patch(*args, referenced_output_features: Optional[Dict[str, TENSOR_OR_LIST_OF_TENSORS_T]] = None, **kwargs):
                        return candidate.forward(*args, **kwargs)
                    candidate.forward = _forward_monkey_patch
                candidates[candidate_name] = candidate
            if len(candidate_refs) > 0:
                # Candidates tensor references are agregated and stored in parent LayerChoice so that `self._submodule_references` only stores references of top-level submodules (not nested candidates)
                module.referenced_submodules = sum(candidate_refs, tuple())

            # Instanciate NNI NAS Mutable LayerChoice from parsed candidates and parameters
            module = nni_mutables.LayerChoice(op_candidates=candidates, reduction=reduction, return_mask=return_mask, key=submodule_name)
        # Create submodule (nested DeepcvModule or submodule from `submodule_creators` or `torch.nn.Module` submodule)
        elif submodule_type == self.NEW_BRANCH_FROM_TENSOR_TOKEN:
            # Similar to dense links but will only use referenced submodule(s) output, allowing new siamese/parrallel NN branches to be defined (wont reuse previous submodule output features)
            if self.FROM_TOKEN not in params and self.FROM_NNI_MUTABLE_INPUT_TOKEN not in params:
                raise ValueError(f'Error: You must either specify "{self.FROM_TOKEN}" or "{self.FROM_NNI_MUTABLE_INPUT_TOKEN}" parameter in a "link" submodule spec.')
            module = ForwardCallbackSubmodule(_new_branch_from_tensor_forward)
            self._setup_forward_callback_submodule(submodule_name, submodule_params=params, forward_callback_module=module)
        else:
            # Parses a regular NN submodule from specs. (either based on a submodule creator or directly a `torch.nn.Module` type or string identifier)
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
                # Specified submodule is assumed to be directly a `torch.nn.Module` or `Callable[..., torch.nn.Module]` type which will be instanciated with its respective parameters as possible arguments according to its `__init__` signature (`params` and global NN spec. parameters)
                fn_or_type = submodule_type

            # Create layer/block submodule from its module_creator or its `torch.nn.Module.__init__()` method (`fn_or_type`)
            submodule_signature_params = inspect.signature(fn_or_type).parameters
            provided_params = {n: v for n, v in params_with_globals.items() if n in submodule_signature_params}
            submdule_params = dict(**params)
            for n in provided_params.keys():
                if n in submdule_params:
                    # Avoid to provide the same parameter twice (either provided through `submdule_params` dict or directly as an argument named after this parameter `n`)
                    del submdule_params[n]
            provided_params.update({n: v for n, v in {'submodule_params': submdule_params, 'prev_shapes': self._features_shapes}.items() if n in submodule_signature_params})
            module = fn_or_type(**provided_params)

            # Process submodule creators output `torch.nn.Module` so that `ForwardCallbackSubmodule` submodules instances are handled in a specific way for output tensor references (e.g. dense/residual), NNI NAS Mutable InputChoice and reduction function builtin support. (these modules are defined by forward pass callbacks which may be fed with referenced sub-module(s) output and a reduction function in addition to previous sub-module output)
            if isinstance(module, ForwardCallbackSubmodule):
                # Submodules which are instances of `ForwardCallbackSubmodule` are handled sperately allowing builtin reduction function, output tensor (residual/dense) references and NNI NAS Mutable InputChoice support. (`DeepcvModule`-specific `torch.nn.Module` defined from a callback called on forward passes)
                self._setup_forward_callback_submodule(submodule_name, submodule_params=params, forward_callback_module=module)
            elif not isinstance(module, torch.nn.Module):
                raise RuntimeError(f'Error: Wrong sub-module creator function/class __init__ return type '
                                   f'(must either be a torch.nn.Module or a `forward` callback of type: `{MODULE_CREATOR_FORWARD_CALLBACK_T}`.')

        return submodule_name, module

    def _submodule_name_and_params_from_spec(self, submodule_spec: Union[Dict, Type[torch.nn.Module], str, Callable[..., torch.nn.Module]], submodule_name: str, existing_submodule_names: Sequence[str]) -> Tuple[str, Dict, Union[Type[torch.nn.Module], str, Callable[..., torch.nn.Module]]]:
        # Retreive submodule parameters and type for all possible submodule spec. senarios
        submodule_type, params = list(submodule_spec.items())[0] if isinstance(submodule_spec, Dict) else (submodule_spec, {})
        if isinstance(params, List) or isinstance(params, Tuple):
            # Architecture definition specifies a sub-module name explicitly
            submodule_name, params = params[0], params[1]
        elif isinstance(params, str):
            # Architecture definition specifies a sub-module name explicitly without any other sub-module parameters
            submodule_name, params = params, dict()
        elif isinstance(params, Dict[str, Any]) and self.SUBMODULE_NAME_TOKEN in params:
            # Allow to specify submodule name in submodule parameters dict instead of throught Tuple/List usage (`DeepcvModule.SUBMODULE_NAME_TOKEN` parameter)
            submodule_name = params[self.SUBMODULE_NAME_TOKEN]
            del params[self.SUBMODULE_NAME_TOKEN]

        # Checks whether if submodule_name is invalid or duplicated
        if submodule_name in existing_submodule_names or submodule_name == r'' or not isinstance(submodule_name, str):
            raise ValueError(f'Error: Invalid or duplicate sub-module name/label: "{submodule_name}"')
        # Checks if `params` is a valid
        if not isinstance(params, Dict):
            raise RuntimeError(f'Error: Architecture sub-module spec. must either be a parameters Dict, or a submodule name along with parameters Dict, but got: "{params}".')

        return (submodule_name, params, submodule_type)

    def _setup_forward_callback_submodule(self, submodule_name: str, submodule_params: Dict[str, Any], forward_callback_module: ForwardCallbackSubmodule) -> Tuple[str, Optional[torch.nn.Module]]:
        """ Specfic model definition logic for submodules based on forward pass callbacks (`ForwardCallbackSubmodule` submodule instances are handled sperately allowing builtin reduction function, output tensor (residual/dense) references and NNI NAS Mutable InputChoice support).
        Allows referencing other submodule(s) output tensor(s) (`DeepcvModule.FROM_TOKEN` usage), NNI NAS Mutable InputChoice (`DeepcvModule.FROM_NNI_MUTABLE_INPUT_TOKEN` usage) and reduction functions (`DeepcvModule.REDUCTION_FUNCTION_TOKEN` usage).
        """
        self._uses_forward_callback_submodules = True
        # DeepcvModule.FROM_NNI_MUTABLE_INPUT_TOKEN occurences in `submodule_params` are handled like DeepcvModule.FROM_TOKEN entries: nni_mutables.InputChoice(references) + optional parameters 'n_chosen' (None by default, should be an integer between 1 and number of candidates) and 'reduction'
        if self.FROM_NNI_MUTABLE_INPUT_TOKEN in submodule_params:
            self._uses_nni_nas_mutables = True
            n_chosen = submodule_params['n_chosen'] if self.FROM_NNI_INPUT_N_CHOSEN_TOKEN in submodule_params else None
            n_candidates = len(submodule_params[self.FROM_NNI_MUTABLE_INPUT_TOKEN])
            mask = submodule_params[self.NNI_NAS_MUTABLE_RETURN_MASK_TOKEN] if self.NNI_NAS_MUTABLE_RETURN_MASK_TOKEN in submodule_params else False
            forward_callback_module._mutable_input = nni_mutables.InputChoice(n_candidates=n_candidates, n_chosen=n_chosen, return_mask=mask, key=submodule_name, reduction='none')

            if self.FROM_TOKEN in submodule_params:
                raise ValueError(f'Error: Cant both specify "{self.self.FROM_TOKEN}" and "{self.FROM_NNI_MUTABLE_INPUT_TOKEN}" in the same submodule '
                                 '(You should either choose to use NNI NAS Mutable InputChoice candidate reference(s) or regular tensor reference(s)).')
        elif self.NNI_NAS_MUTABLE_RETURN_MASK_TOKEN in submodule_params or self.FROM_NNI_INPUT_N_CHOSEN_TOKEN:
            raise ValueError(f'Error: Cannot specify "{self.NNI_NAS_MUTABLE_RETURN_MASK_TOKEN}" nor "{self.FROM_NNI_INPUT_N_CHOSEN_TOKEN}" without using "{self.FROM_NNI_MUTABLE_INPUT_TOKEN}".'
                             f'("{self.NNI_NAS_MUTABLE_RETURN_MASK_TOKEN}" and "{self.FROM_NNI_INPUT_N_CHOSEN_TOKEN}" is an optional parameter reserved for NNI NAS InputChoice usage).')

        # Store any sub-module name/label references (used to store referenced submodule's output features during model's forward pass in order to reuse these features later in a forward callback (e.g. for residual links))
        if self.FROM_TOKEN in submodule_params or self.FROM_NNI_MUTABLE_INPUT_TOKEN in submodule_params:
            # Allow multiple referenced sub-module(s) (`_from` entry can either be a list/tuple of referenced sub-modules name/label or a single sub-module name/label)
            tensor_references = submodule_params[self.FROM_TOKEN] if self.FROM_TOKEN in submodule_params else submodule_params[self.FROM_NNI_MUTABLE_INPUT_TOKEN]
            forward_callback_module.referenced_submodules = tuple((tensor_references,)) if issubclass(type(tensor_references), str) else tuple(tensor_references)

            if self.REDUCTION_FUNCTION_TOKEN in submodule_params:
                if submodule_params[self.REDUCTION_FUNCTION_TOKEN] not in self.TENSOR_REDUCTION_FUNCTIONS:
                    raise ValueError(f'Error: Tensor reduction function ("_reduction" parameter) should be one these string values: {TENSOR_REDUCTION_FUNCTIONS.keys()}, got `"_reduction"="{submodule_params["_reduction"]}"`. '
                                     '"_reduction" is the reduction function applied to referenced tensors in "_from" (or chosen tensors in "_from_nni_mutable_input" if `"n_chosen" > 1`)')
                forward_callback_module._reduction_function = TENSOR_REDUCTION_FUNCTIONS[submodule_params[self.REDUCTION_FUNCTION_TOKEN]]


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


#__________________________________________ DEFAULT/BASE SUBMODULE CREATORS ___________________________________________#

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


def _residual_dense_link(is_residual: bool = True) -> Callable[['submodule_params', 'allow_scaling', 'scaling_mode', 'channel_dim'], ForwardCallbackSubmodule]:
    """ Creates a residual or dense link sub-module which concatenates or adds features from previous sub-module output with other referenced sub-module(s) output(s).
    `submodule_params` argument must contain a `_from` (or `_from_nni_mutable_input`) entry giving the other sub-module reference(s) (sub-module name(s)) from which output(s) is added to previous sub-module output features.

    Returns a callback which is called during foward pass of DeepcvModule.
    Like any other DeepcvModule submodule creators which returns a forward callback and which uses tensor references (`_from` or `_from_nni_mutable_input`), a reduction function can be specified (see `REDUCTION_FUNCTION_T`) in `_from` or `_from_nni_mutable_input` parameters: by default, reduction function will be 'sum' if this is a residual link and 'concat' if this is a dense link.
    If 'allow_scaling' is `True` (when `allow_scaling: True` is specified in YAML residual/dense link spec.), then if residual/dense features have different shapes on dimensions following `channel_dim`, it will be scaled (upscaled or downscaled) using [`torch.functional.interpolate`](https://pytorch.org/docs/stable/nn.functional.html#interpolate).
    By default interpolation is 'linear'; If needed, you can specify a `scaling_mode` parameter as well to change algorithm used for up/downsampling (valid values: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area', for more details, see [`torch.nn.functional.interpolate` doc](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate)).
    NOTE: Scaling/interpolation of residual/dense tensors is only supported for 1D, 2D and 3D features, without taking into account channel and minibatch dimensions. Also note that `minibatch` dimension is required by [`torch.functional.interpolate`](https://pytorch.org/docs/stable/nn.functional.html#interpolate).
    NOTE: If `allow_scaling` is `False`, output features shapes of these two or more submodules must be the same, except for the channels/filters dimension if this is a dense link.
    NOTE: The only diference between residual and dense links ('is_residual' beeing 'True' of 'False') is the default `DeepcvModule.REDUCTION_FUNCTION_TOKEN` function beeing respectively 'sum' and 'concat'.
    """
    def _create_link_submodule(submodule_params: Dict[str, Any], is_residual: bool, allow_scaling: bool = False, scaling_mode: str = 'linear', channel_dim: int = 1) -> ForwardCallbackSubmodule:
        def _forward_callback(x: TENSOR_OR_LIST_OF_TENSORS_T, referenced_submodules_out: List[torch.Tensor], tensor_reduction_fn: REDUCTION_FUNCTION_T = (TENSOR_REDUCTION_FUNCTIONS['sum'] if is_residual else TENSOR_REDUCTION_FUNCTIONS['concat'])):
            """ Redisual or Dense link forward pass callbacks
            If target output shape is different from one of the referenced tensor shapes, a up/down-scaling (interpolation) may be performed according to `scaling_mode` and `allow_scaling` parameters.
            NOTE: Target output shape is assumed to be the same as `x` features shape if `x` is a `torch.Tensor` or the same as the first tensor shape of `x` if `x` is a list of `torch.Tensor`s.
            A reduction function can be specified (see `TENSOR_REDUCTION_FUNCTIONS`); If this is a residual link, reduction function defaults to 'sum' and if this is a dense link, reduction function defaults to 'concat'.
            """
            tensors = [x, ] if isinstance(x, torch.Tensor) else [*x, ]
            for y in referenced_submodules_out:
                if tensors[0].shape[channel_dim + 1:] != y.shape[channel_dim + 1:]:
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
        return ForwardCallbackSubmodule(_forward_callback)

    _create_link_submodule.__doc__ = _residual_dense_link.__doc__
    return partial(_create_link_submodule, is_residual=is_residual)


def _new_branch_from_tensor_forward(x: TENSOR_OR_LIST_OF_TENSORS_T, referenced_submodules_out: List[torch.Tensor], tensor_reduction_fn: REDUCTION_FUNCTION_T = TENSOR_REDUCTION_FUNCTIONS['concat'], channel_dim: int = 1):
    """ Simple forward pass callback which takes referenced output tensor(s) and ignores previous submodule output features, allowing to define siamese/parallel branches thereafter.
    In other words, `DeepcvModule.NEW_BRANCH_FROM_TENSOR_TOKEN` submodules are similar to dense links but will only use referenced submodule(s) output, allowing new siamese/parrallel NN branches to be defined (wont reuse previous submodule output features)
    If multiple tensors are referenced using `DeepcvModule.FROM_TOKEN` (or `DeepcvModule.FROM_NNI_MUTABLE_INPUT_TOKEN`), `tensor_reduction_fn` reduction function will be applied.
    Reduction function is 'concat' by default and can be overriden by `_reduction` parameter in link submodule spec., see `TENSOR_REDUCTION_FUNCTIONS` for all possible reduction functions.
    NOTE: As `DeepcvModule.NEW_BRANCH_FROM_TENSOR_TOKEN` submodules have a specific handling/meaning in `DeepcvModule`, this callback is directly used in `DeepcvModule.define_nn_architecture` instead of beeing part of a regular submodule creator in `BASIC_SUBMODULE_CREATORS` (like `_deepcvmodule` or `_nni_mutable_layer` submodules).
    """
    # Ignores `x` input tensor (previous submodule output tensor is ignored)
    return tensor_reduction_fn(referenced_submodules_out, dim=channel_dim)


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
