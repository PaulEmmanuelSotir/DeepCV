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
import torch.nn

import numpy as np
import nni
import nni.nas.pytorch.mutables as nni_mutables

from deepcv.utils import NL, human_readable_size, import_tests
import deepcv.meta.nn
import deepcv.meta.nn_spec
import deepcv.meta.submodule_creators
from deepcv.meta.types_aliases import *


__all__ = ['DeepcvModule', 'DeepcvModuleWithSharedImageBlock', 'DeepcvModuleDescriptor']
__author__ = 'Paul-Emmanuel Sotir'

#_______________________________________________ DEEPCVMODULE CLASSES _________________________________________________#


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

    `DeepcvModule` have 'global' Weight Norm and Spectral Norm support through 'weight_norm' and 'spectral_norm' parameters dicts of `hp`:  
        - 'weight_norm' parameter dict may specify 'name' and 'dim' arguments, see [Weight Norm in PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html?highlight=weight%20norm#torch.nn.utils.weight_norm)
        - 'spectral_norm' parameter dict may specify 'name', 'n_power_iterations', 'eps' and 'dim' arguments, see [Spectral Norm in PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html?highlight=spectral#torch.nn.utils.spectral_norm)
    NOTE: If needed, you can remove weight norm or spectral norm hooks afterwards by using: `torch.nn.utils.remove_weight_norm(module, name=...)` or `torch.nn.utils.remove_spectral_norm(module, name=...)`  

    .. For more details about `architecture` hyperparameter parsing, see code in `deepcv.meta.nn_spec.define_nn_architecture` and examples of DeepcvModule(s) YAML architecture specification in ./conf/base/parameters.yml
    NOTE: A sub-module's name defaults to 'submodule_{i}' where 'i' is sub-module index in architecture sub-module list. Alternatively, you can specify a sub-module's name in architecture configuration (either using a tuple/list of submodule name and params or by specifying a `_name` argument in submodule params), which, for example, allows you to define residual/dense links referencing these tensor(s) by their name(s).
    .. See complete examples of `DeepcvModule` model NN architecture specification in `[Kedro hyperparameters YAML config file]conf/base/parameters.yml`
    """

    HP_DEFAULTS = {'architecture': ..., 'act_fn': ..., 'weight_norm': None, 'spectral_norm': None}

    def __init__(self, input_shape: torch.Size, hp: HYPERPARAMS_T, additional_submodule_creators: Optional[Dict[str, Union[Callable, Type[torch.nn.Module]]]] = None, extend_basic_submodule_creators_dict: bool = True, additional_init_logic: Callable[[torch.nn.Module, 'xavier_gain'], None] = None):
        """ Instanciates a new `DeepcvModule` instance

        Args:
            - input_shape: PyTorch Tensor shape which will be taken by this model during forward passes.
            - hp: Hyperparameters map (Dict or `deepcv.meta.hyperparams.Hyperparameters`) which should/could have required/optional arguments specified in `self.HP_DEFAULTS`. `architecture` hyperparameter should contain NN architecture specification, probably coming from `./conf/base/parameters.yml` YAML parameters (see `deepcv.meta.nn_spec.define_nn_architecture` for more details on parsing of NN architecture specs.).  
            - additional_submodule_creators: Dict of possible architecture sub-modules associated with their respective module creators. If None, then defaults to `deepcv.meta.submodule_creators.BASIC_SUBMODULE_CREATORS`.
            - extend_basic_submodule_creators_dict: Boolean indicating whether `submodule_creators` argument will be extended with `deepcv.meta.submodule_creators.BASIC_SUBMODULE_CREATORS` dict or not. i.e. whether `submodule_creators` defines additionnal sub-modules or all existing sub-modules.  
                NOTE: If `True` and some creator name(s)/entries are both present in `submodule_creators` arg and in `deepcv.meta.submodule_creators.BASIC_SUBMODULE_CREATORS`, then `submodule_creators` dict values overrides defaults/base ones.  
            - additional_init_logic: Callback which may be used during parameter(s) initialization: Will be called for any unsupported child module(s) which have parameters to be initialized. This callback allows you to extend `self._initialize_parameters` function support for `torch.nn.Module`(s) other than convolutions, `torch.nn.Linear` and BatchNorm*d modules. This function should also take a `xavier_gain` argument along with `module` to be initialized.
        """
        from deepcv.meta.hyperparams import to_hyperparameters
        super().__init__()
        self._input_shape = input_shape
        self._uses_nni_nas_mutables = False
        self._uses_forward_callback_submodules = False
        self._additional_init_logic = additional_init_logic

        # Process module hyperparameters
        assert self.HP_DEFAULTS != ..., f'Error: Module classes which inherits from "DeepcvModule" ({type(self).__name__}) must define "HP_DEFAULTS" class attribute dict.'
        self._hp, _missing = to_hyperparameters(hp, defaults=self.HP_DEFAULTS, raise_if_missing=True)

        # Create model architecture according to hyperparameters's `architecture` entry (see ./conf/base/parameters.yml for examples of architecture specs.) and initialize its parameters
        deepcv.meta.nn_spec.define_nn_architecture(self, self._hp['architecture'], submodule_creators=additional_submodule_creators,
                                                   extend_basic_submodule_creators_dict=extend_basic_submodule_creators_dict)
        self._initialize_parameters(self._hp['act_fn'], additional_init_logic=self._additional_init_logic)

        # Support for weight normalization and spectral weight normalization
        if self._hp['weight_norm'] is not None:
            # Weight Norm: Implemented with hooks added on given module in order to apply normalization on parameter(s) named after `self._hp['weight_norm']['name']` (defaults to 'weight')
            torch.nn.utils.weight_norm(self, **self._hp['weight_norm'])
        elif self._hp['spectral_norm'] is not None:
            # Spectral Norm: Weight norm with spectral norm of the weight matrix calculated using power iteration method (Implemented with hooks added on given module)
            torch.nn.utils.spectral_norm(self, **self._hp['spectral_norm'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == len(self._input_shape):
            # Turn single input tensor into a batch of size 1 (i.e. make sure there is a minibatch dimension)
            x = x.unsqueeze(dim=0)

        # Apply DeepcvModule neural net architecture defined from a parsed spec. in `deepcv.meta.nn_spec.define_nn_architecture`
        if not self.is_sequencial_nn():
            # DeepcvModule NN architecure cant be applied with a single `torch.nn.Sequential`: secific handling is needed during forwarding due to output tensor reference(s), `deepcv.meta.nn_spec.yaml_tokens.NEW_BRANCH_FROM_TENSOR` submodule usage, and/or builtin reduction function support.
            referenced_output_features = {}
            # Stores remaining submodules output tensor references (shallow copy to avoid removing entries from `self._submodule_references` when reference(s) are consumed during forward pass/propagation)
            remaining_submodule_references = copy.copy(self._submodule_references)

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

                if isinstance(subm, nni_mutables.LayerChoice) or isinstance(subm, deepcv.meta.submodule_creators.ForwardCallbackSubmodule):
                    # Forward pass through a `deepcv.meta.submodule_creators.ForwardCallbackSubmodule` or NNI NAS LayerChoice submodule (forward pass throught one or more underlying candidate submodules)
                    x = subm(x, referenced_output_features=current_subm_references)
                else:
                    # Forward pass through regular `torch.nn.Module` submodule
                    x = subm(x)

                # If submodule is referenced by another module by a `deepcv.meta.nn_spec.yaml_tokens.FROM` entry in its parameters, then we store its output features for later use (e.g. for a residual/dense link)
                if name in sum(remaining_submodule_references.values(), tuple()):
                    referenced_output_features[name] = x
                return x
        elif hasattr(self, '_sequential_net'):
            # NN architecture is only a `torch.nn.Sequential` model (no submodules which makes usage of tensor references, reduction functions nor NNI NAS Mutables, i.e., no `deepcv.meta.nn_spec.yaml_tokens.FROM` nor `deepcv.meta.nn_spec.yaml_tokens.FROM_NAS_INPUT_CHOICE` usage, ect...)
            return self._sequential_net(x)
        else:
            raise RuntimeError('Error: Incoherent `DeepcvModule` model; No `_sequential_net` attribute while NN architecture seams to be sequential'
                               '(no residual/dense link usage, nor NNI NAS MutableInput usage, nor builtin reduction function specified, ...). There may be a bug in model definition from NN architecture specification in `deepcv.meta.nn_spec.define_nn_architecture`')

    def __str__(self) -> str:
        """ Describes DeepCV module in a human readable text string, see `DeepcvModule.describe()` function or `DeepcvModuleDescriptor` class for more details """
        return str(self.describe())

    def uses_nni_nas_mutables(self, recursive: bool = False) -> bool:
        """ Returns a boolean which indicates whether `DeepcvModule` instance makes usage of NNI NAS Mutable API (`LayerChoice`(s) or `InputChoice`(s) throuht usgae of `NAS_LAYER_CHOICE` or `FROM_NAS_INPUT_CHOICE` submodule of YAML architecture spec.) 
        NOTE: `uses_nni_nas_mutables()` may be `False` even if a nested `DeepcvModule` (`deepcv.meta.nn_spec.yaml_tokens.NESTED_DEEPCV_MODULE` spec) does makes usage of NNI NAS Mutable API, specify `recursive=True` if you need to know about usage of NNI NAS Mutable API even in nested `DeepcvModule`(s)
        """
        if not recursive:
            return self._uses_nni_nas_mutables
        else:
            return any([subm.uses_nni_nas_mutables(recursive=True) for subm in self._submodules.values() if isinstance(subm, DeepcvModule)])

    def uses_forward_callback_submodules(self, recursive: bool = False) -> bool:
        """ Returns a boolean which indicates whether `DeepcvModule` instance makes usage of NNI NAS Mutable API (`LayerChoice`(s) or `InputChoice`(s) throuht usgae of `NAS_LAYER_CHOICE` or `FROM_NAS_INPUT_CHOICE` submodule of YAML architecture spec.)
        NOTE: `uses_forward_callback_submodules()` may be `False` even if a nested `DeepcvModule` (`deepcv.meta.nn_spec.yaml_tokens.NESTED_DEEPCV_MODULE` spec) does makes usage of `deepcv.meta.submodule_creators.ForwardCallbackSubmodule` submodule(s), specify `recursive=True` if you need to know about usage of `deepcv.meta.submodule_creators.ForwardCallbackSubmodule` even in nested `DeepcvModule`(s)
        """
        if not recursive:
            return self._uses_forward_callback_submodules
        else:
            return any([subm.uses_forward_callback_submodules(recursive=True) for subm in self._submodules.values() if isinstance(subm, DeepcvModule)])

    def is_sequencial_nn(self, recursive: bool = False) -> bool:
        """ Returns a boolean indicating whether `DeepcvModule` NN architecture (parsed from YAML NN specs.) can be applied as a `torch.nn.Sequential` (`self._sequential_net` attribute then exists and is a `torch.nn.Sequential`).
        I.e., returns whether if `DeepcvModule` architecture doesn't uses any residual/dense links (Or any other `ForwardCallbackSubmodule`), nor `deepcv.meta.nn_spec.yaml_tokens.NEW_BRANCH_FROM_TENSOR` submodules, nor NNI NAS mutable inputs/layers, nor builtin reduction function support which all needs specific handling during forward passes (impossible to apply whole architecture by using a single `torch.nn.Sequential`). """
        return not self.uses_nni_nas_mutables(recursive=recursive) and not self.uses_forward_callback_submodules(recursive=recursive)

    def describe(self):
        """ Describes DeepCV module with its architecture, capacity and features shapes at sub-modules level.
        Args:
            - to_string: Whether deepcv NN module should be described by a human-readable text string or a NamedTuple of various informations which, for example, makes easier to visualize model's sub-modules capacities or features shapes...
        Returns a `DeepcvModuleDescriptor` which contains model name, capacity, and eventually submodules names, feature shapes/dims/sizes and capacities...
        """
        return DeepcvModuleDescriptor(self)

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

    def _initialize_parameters(self, act_fn: Type[torch.nn.Module] = None, additional_init_logic: Callable[[torch.nn.Module, 'xavier_gain'], None] = None):
        """ Initializes model's parameters with Xavier Initialization with a scale depending on given activation function (only needed if there are convolutional and/or fully connected layers).
        NOTE: For now this function only support Xavier Init for Linear (fc), convolutions and BatchNorm*d parameters/weights and Xavier Init gain will be calculated once from global `act_fn` values and won't take into account any different/overriding activation functions specified in a submodule spec. (See `additional_init_logic` argument if better intialization support is needed)  
        Args:
            - act_fn: Activation function type which is used to process Xavier Init gain. (NOTE: All submodules/layers are assumed to use this same activation function and will use the smae Xavier Init gain)
            - additional_init_logic: Function called with all unsupported modules which have parameters to be initialized, allowing to extend initialization support for `torch.nn.Module`s other than convolutions, `torch.nn.Linear` and BatchNorm*d modules. This function should also take an `xavier_gain` (Optional[float]) argument along with `module` to be initialized.
        """
        xavier_gain = torch.nn.init.calculate_gain(deepcv.meta.nn.get_gain_name(act_fn)) if act_fn else None

        def _raise_if_no_act_fn(sub_module_name: str):
            # NOTE: This function is needed to avoid raising if activation function isn't needed (no xavier init to be performed if there isn't any conv nor fully connected parameters/weights)
            if xavier_gain is None:
                raise ValueError(f'Error: Must specify `act_fn` argument in `DeepcvModule._initialize_parameters` function in order to initialize '
                                 f'{sub_module_name} sub-module(s) with Xavier Init. (See `deepcv.meta.nn.get_gain_name` for supported activation functions)')

        def _xavier_init(module: torch.nn.Module, additional_init: Callable[[torch.nn.Module, 'xavier_gain'], None]):
            if deepcv.meta.nn.is_conv(module):
                _raise_if_no_act_fn('convolution')
                torch.nn.init.xavier_normal_(module.weight.data, gain=xavier_gain)
                module.bias.data.fill_(0.)
            elif deepcv.meta.nn.is_fully_connected(module):
                _raise_if_no_act_fn('fully connected')
                torch.nn.init.xavier_uniform_(module.weight.data, gain=xavier_gain)
                module.bias.data.fill_(0.)
            elif type(module).__module__ == torch.nn.BatchNorm2d.__module__:  # `torch.nn.modules.batchnorm` PyTorch Module as of PyTorch 1.6.0
                torch.nn.init.uniform_(module.weight.data)  # gamma == weight here
                module.bias.data.fill_(0.)  # beta == bias here
            elif len(list(module.parameters(recurse=False))) > 0:
                if additional_init is None:
                    raise TypeError(f'ERROR: Some module(s) which have parameter(s) cant be explicitly initialized by `DeepcvModule._initialize_parameters`: '
                                    f'"{type(module)}" `torch.nn.Module` type not supported.{NL}'
                                    'Note you can extend `DeepcvModule._initialize_parameters` initialization strategy by providing `additional_init_logic` arg when instanciating `DeepcvModule`.')
                else:
                    additional_init(module, xavier_gain=xavier_gain)
        self.apply(partial(_xavier_init, additional_init=additional_init_logic))


class DeepcvModuleWithSharedImageBlock(DeepcvModule):
    """ Deepcv Module With Shared Image Block model base class
    Appends to DeepcvModule a basic shared convolution block allowing transfert learning between all DeepCV models on images.
    """

    SHARED_BLOCK_DISABLED_WARNING_MSG = r'Warning: `DeepcvModule.{}` called while `self._enable_shared_image_embedding_block` is `False` (Shared image embedding block disabled for this model)'

    def __init__(self, input_shape: torch.Size, hp: 'deepcv.meta.hyperparams.Hyperparameters', enable_shared_block: bool = True, freeze_shared_block: bool = True):
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

    @classmethod
    def _define_shared_image_embedding_block(cls, in_channels: int = 3, channel_dim: int = 1):
        logging.info(f'Creating shared image embedding block of `{cls.__name__}` models...')
        norm_args = {'affine': True, 'eps': 1e-05, 'momentum': 0.0736}
        # ``input_shape`` argument doesnt needs valid dimensions apart from channel dim as we are using BatchNorm (See `deepcv.meta.nn.normalization_techniques` documentation)
        def _norm_ops(chnls): return dict(norm_type=deepcv.meta.nn.NormTechnique.BATCH_NORM, norm_kwargs=norm_args, input_shape=[chnls, 10, 10])

        layers = [('shared_block_conv_1', deepcv.meta.nn.layer(layer_op=torch.nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), padding=1),
                                                               act_fn=torch.nn.ReLU, **_norm_ops(8))),
                  ('shared_block_conv_2', deepcv.meta.nn.layer(layer_op=torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
                                                               act_fn=torch.nn.ReLU, **_norm_ops(16))),
                  ('shared_block_conv_3', deepcv.meta.nn.layer(layer_op=torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1),
                                                               act_fn=torch.nn.ReLU, **_norm_ops(8))),
                  ('shared_block_conv_4', deepcv.meta.nn.layer(layer_op=torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), padding=1),
                                                               act_fn=torch.nn.ReLU, **_norm_ops(4)))]
        cls.shared_image_embedding_block = torch.nn.Sequential(OrderedDict(*layers))


class DeepcvModuleDescriptor:
    """ Describes DeepCV module with its architecture, capacity and features shapes at sub-modules level """

    def __init__(self, module: DeepcvModule):
        self.module = module

        if hasattr(module, '_architecture_spec'):
            # NOTE: `module.architecture_spec` attribute will be defined if `deepcv.meta.nn_spec.define_nn_architecture` is called
            architecture_spec = module._architecture_spec
        elif 'architecture' in module._hp:
            # otherwise, we try to look for architecture/sub-modules configuration in hyperparameters dict
            architecture_spec = module._hp['architecture']
        else:
            architecture_spec = None
            logging.warn(f"Warning: `{type(self).__name__}({type(module)})`: cant find NN architecture, no `module.architecture_spec` attr. nor `architecture` in `module._hp`")

        # Fills and return a DeepCV module descriptor
        self.capacity = deepcv.meta.nn.get_model_capacity(module)
        self.human_readable_capacity = human_readable_size(self.capacity)
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
            self.human_readable_capacities = map(human_readable_size, module._submodules_capacities)

    def __str__(self) -> str:
        """ Ouput a human-readable string representation of the deepcv module based on its descriptor """
        if self.architecture is not None:
            features = self.submodules_features_shapes if hasattr(self, 'submodules_features_shapes') else ['UNKNOWN'] * len(self.architecture)
            capas = self.human_readable_capacities if hasattr(self, 'human_readable_capacities') else ['UNKNOWN'] * len(self.architecture)
            desc_str = f'{NL}\t'.join([f'- {n}({p}) output_features_shape={s}, capacity={c}' for (n, p), s, c in zip(self.submodules.items(), features, capas)])
        else:
            desc_str = '(No submodule architecture informations to describe)'

        if isinstance(self.module, DeepcvModuleWithSharedImageBlock):
            desc_str += f'{NL} SIEB (Shared Image Embedding Block) usage:'
            if self.uses_shared_block:
                desc_str += 'This module makes use of shared image embedding block applied to input image:'
                if self.did_forked_shared_block:
                    desc_str += f"{NL}\t- FORKED=True: Shared image embedding block parameters have been forked, SGD updates from other models wont impact this model's weights and SGD updates of this model wont change shared weights of other models until the are eventually merged"
                else:
                    desc_str += f'{NL}\t- FORKED=False: Shared image embedding block parameters are still shared, any non-forked/non-freezed DeepcvModule SGD uptates will have an impact on these parameters.'
                if self.freezed_shared_block:
                    desc_str += f'{NL}\t- SHARED=True: Shared image embedding block parameters have been freezed and wont be taken in account in gradient descent training of this module.'
                else:
                    desc_str += f'{NL}\t- SHARED=False: Shared image embedding block parameters are not freezed and will be learned/fine-tuned during gradient descent training of this model.'
            else:
                desc_str = ' This module doesnt use shared image embedding block.'
        return f'{self.model_class_name} (capacity={self.human_readable_capacity}):{NL}\t{desc_str}'

#______________________________________________ DEEPCV MODULE UNIT TESTS ______________________________________________#


if __name__ == '__main__':
    cli = import_tests().test_module_cli(__file__)
    cli()
