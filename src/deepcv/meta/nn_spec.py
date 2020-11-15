#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Neural Network Architecture specification meta module - nn_spec.py - `DeepCV`__
Neural Network Architecture specification parsing meta module for base_module.DeepcvModule NN definition from YAML specs
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import copy
import enum
import inspect
import functools
from collections import OrderedDict
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence, List

import torch
import torch.nn
import torch.nn.functional

import nni
import nni.nas.pytorch.mutables as nni_mutables

import deepcv.utils
from .types_aliases import *
from .nn import get_model_capacity, get_out_features_shape


__all__ = ['DEFAULT_LAYER_CHOICE_REDUCTION', 'yaml_tokens', 'define_nn_architecture']
__author__ = 'Paul-Emmanuel Sotir'

""" Default reduction function of NNI NAS Mutable `LayerChoices`
NOTE: Unlike NNI NAS `InputChoice`, NNI NAS `LayerChoice`(s) uses NNI builtin reduction function mechanism instead of DeepCV's `TENSOR_REDUCTION_FNS`/`get_reduction_fn`; However, both supports the same reduction function names.
"""
DEFAULT_LAYER_CHOICE_REDUCTION = r'mean'


class yaml_tokens(enum.Enum):
    """ 'Enum' class storing special tokens which can be used in YAML architecture specification of `deepcv.meta.base_module.DeepcvModule` modules
    Those are builtin YAML spec tokens used by `DeepcvModule` NN architecture definition implementation (builtin means here without being submodule creators)
    """
    FROM = r'_from'
    SUBMODULE_NAME = r'_name'
    NAS_LAYER_CHOICE = '_nas_layer_choice'
    NESTED_DEEPCV_MODULE = r'_nested_deepcv_module'
    FROM_NAS_INPUT_CHOICE = r'_from_nas_input_choice'
    # `NAS_LAYER_REDUCTION_FN` NNI LAYER CHOICE REDUCTION FN (Only used for NNI builtin reduction function usage for NNI NAS LayerChoice(s); I.e. unrelated to `TENSOR_REDUCTION_FNS`/`get_reduction_fn` nor other submodules which make use of DeepCV reduction functions)
    NAS_LAYER_REDUCTION_FN = r'_reduction'
    # NOTE: Unlike other `DeepcvModule`-builtin tokens in this Enum, `NEW_BRANCH_FROM_TENSOR` is used for a submodule which is implemented like any other submodule creators with an entry in `deepcv.meta.submodule_creators.BASIC_SUBMODULE_CREATORS` (Ie. even if `NEW_BRANCH_FROM_TENSOR` is a builtin token of `DeepcvModule`, this submodule can be disabled by giving a non-default submodule creators dict or by removing its entry in `deepcv.meta.submodule_creators.BASIC_SUBMODULE_CREATORS`)
    NEW_BRANCH_FROM_TENSOR = '_new_branch_from_tensor'
    FROM_NAS_INPUT_N_CHOSEN = r'_n_chosen'
    NAS_MUTABLE_RETURN_MASK = r'_return_mask'
    NAS_LAYER_CHOICE_CANDIDATES = r'_candidates'

#_____________________________________________ NN SPEC PARSING FUNCTIONS ______________________________________________#


def define_nn_architecture(deepcv_module: 'deepcv.meta.base_module.DeepcvModule', architecture_spec: Iterable, subm_creators: SUBMODULE_CREATORS_DICT_T = None, extend_basic_subm_creators_dict: bool = True):
    """ Defines neural network architecture by parsing 'architecture' hyperparameter and creating sub-modules accordingly
    .. For examples of DeepcvModules YAML architecture specification, see ./conf/base/parameters.yml
    NOTE: defines `deepcv_module._features_shapes`, `deepcv_module._submodules_capacities`, `deepcv_module._submodules` and `deepcv_module._architecture_spec`, ... attributes (usefull for forward passes, debuging and `deepcv_module.__str__` and `deepcv_module.describe` functions)
    TODO: Refactor these spec parsing function to be less tied to DeepcvModule class and make a 'ArchitectureSpecParser' class which stores parsing context instead 
    Args:
        - deepcv_module: `deepcv.meta.base_module.DeepcvModule` NN model to be parsed/defined by YAML architecture specification
        - architecture_spec: Neural net architecture definition listing submodules to be created with their respective parameters (probably from hyperparameters of `conf/base/parameters.yml` configuration file)
        - subm_creators: Dict of possible architecture sub-modules associated with their respective module creators. If None, then defaults to `deepcv.meta.subm_creators.BASIC_SUBMODULE_CREATORS`.
        - extend_basic_subm_creators_dict: Boolean indicating whether `subm_creators` argument will be extended with `deepcv.meta.submodule_creators.BASIC_SUBMODULE_CREATORS` dict or not. i.e. whether `subm_creators` defines additionnal sub-modules or all existing sub-modules.  
            NOTE: If `True` and some creator name(s)/entries are both present in `subm_creators` arg and in `deepcv.meta.submodule_creators.BASIC_SUBMODULE_CREATORS`, then `subm_creators` dict values overrides defaults/base ones.  
    """
    from .submodule_creators import ForwardCallbackSubmodule, BASIC_SUBMODULE_CREATORS
    deepcv_module._features_shapes = [deepcv_module._input_shape, ]
    deepcv_module._architecture_spec = architecture_spec
    deepcv_module._submodules_capacities = list()
    deepcv_module._submodules = OrderedDict()
    # `_submodule_references` is a dict which associates referenced sub-modules name/label with a List of their respective referrer sub-modules name/label (referenced tensor(s) using yaml_tokens.FROM or referenced tensors candidate(s)) using yaml_tokens.FROM_NAS_INPUT_CHOICE)
    deepcv_module._submodule_references: Dict[str, List[str]] = dict()

    # Merge given submodule creators (if any) with defaults (`deepcv.meta.submodule_creators.BASIC_SUBMODULE_CREATORS`) ones if `extend_basic_subm_creators_dict` is `True` (and make sure given `subm_creators` overrides any `BASIC_SUBMODULE_CREATORS` entries/creators which have the same name)
    subm_creators = {**(BASIC_SUBMODULE_CREATORS if extend_basic_subm_creators_dict else dict()),
                     **(subm_creators if subm_creators is not None else dict())}

    # Parse submodule NN architecture spec in order to define PyTorch model's submodules accordingly
    for i, submodule_spec in enumerate(architecture_spec):
        # Parse submodule specification to obtain a new `torch.nn.Module` submodule of NN architecture
        subm_name, subm = _parse_torch_module_from_submodule_spec(deepcv_module, submodule_spec, i, subm_creators)
        # Append new submodule to NN architecture (`deepcv_module._submodules` Ordered Dict)
        deepcv_module._submodules[subm_name] = subm
        # Store tensor references for easier/better memory handling during forward passes (e.g. residual/dense links, `yaml_tokens.NEW_BRANCH_FROM_TENSOR` usage, ...)
        if (isinstance(subm, ForwardCallbackSubmodule) or isinstance(subm, nni_mutables.LayerChoice)) and getattr(subm, 'referenced_submodules') is not None:
            deepcv_module._submodule_references[subm_name] = subm.referenced_submodules
        # Figure out new NN submodule capacity
        deepcv_module._submodules_capacities.append(get_model_capacity(subm))
        if getattr(deepcv_module, '_child_modules', None) is not None:
            deepcv_module._child_modules.clear()  # Reset child modules of DeepcvModule before redefining them as either a `torch.nn.Sequential` or a `torch.nn.ModuleDict`
        if deepcv_module.is_sequential_nn():
            # `torch.nn.Sequential` usage is only accurate/applicable/defined when there isn't any submodules using tensor reference(s), NNI NAS Mutable InputChoice(s)
            deepcv_module._child_modules = torch.nn.Sequential(deepcv_module._submodules)
        else:
            deepcv_module._child_modules = torch.nn.ModuleDict(deepcv_module._submodules)
        # Make sure all referenced sub-module exists (i.e. that there is a matching submodule name/label)
        missing_refs = [ref for ref in deepcv_module._submodule_references[subm_name] if ref not in deepcv_module._submodules.keys()]
        if len(missing_refs) > 0:
            raise ValueError(f'Error: Invalid sub-module reference(s), cant find following sub-module name(s)/label(s): "{missing_refs}".'
                             ' Output tensor references must refer to a previously defined sub-module name.')
        # Figure out output features shape from new submodule by performing a dummy forward pass of `DeepcvModule` instance
        subm_out_shape = get_out_features_shape(deepcv_module._input_shape, deepcv_module, use_minibatches=True)
        deepcv_module._features_shapes.append(subm_out_shape[1:])  # subm_out_shape[1:] removes the first dim as `get_out_features_shape` returns output shape with minibatch dim


def _parse_torch_module_from_submodule_spec(deepcv_module: 'deepcv.meta.base_module.DeepcvModule', submodule_spec: Union[Dict, Type[torch.nn.Module], str, Callable[..., torch.nn.Module]], submodule_pos: Union[int, str], subm_creators: SUBMODULE_CREATORS_DICT_T, default_submodule_prefix: str = '_submodule_', allow_mutable_layer_choices: bool = True) -> Tuple[str, torch.nn.Module]:
    """ Defines a single submodule of `DeepcvModule` from its respective NN architecture spec.  
    TODO: Refactor `_parse_torch_module_from_submodule_spec` and underlying function to be independent of DeepcvModule (and, for ex., put those in a speratate `yaml_nn_spec_parsing.py` file) (Context which still depends on DeepcvModule instance: `deepcv_module._uses_nni_nas_mutables deepcv_module._uses_forward_callback_submodules deepcv_module._features_shapes deepcv_module._hp deepcv_module.HP_DEFAULTS`)  
    """
    from deepcv.meta.submodule_creators import ForwardCallbackSubmodule
    subm_name = default_submodule_prefix + str(submodule_pos)
    subm_name, params, subm_type = _subm_name_and_params_from_spec(submodule_spec, default_subm_name=subm_name, existing_subm_names=deepcv_module._submodules.keys())

    # Add global (hyper)parameters from `hp` to `params` (allows to define parameters like `act_fn`, `dropout_prob`, `batch_norm`, ... either globaly in `hp` or localy in `params` from submodule specs)
    # NOTE: In case a parameter is both specified in `deepcv_module._hp` globals and in `params` local submodule specs, `params` entries from submodule specs will allways override parameters from `hp`
    params_with_globals = {n: copy.deepcopy(v) for n, v in deepcv_module._hp.items() if n not in params}
    params_with_globals.update(params)

    if subm_type == yaml_tokens.NESTED_DEEPCV_MODULE:
        # Allow nested DeepCV sub-module (see deepcv/conf/base/parameters.yml for examples)
        module = type(deepcv_module)(input_shape=deepcv_module._features_shapes[-1], hp=params_with_globals, additional_subm_creators=subm_creators,
                                     extend_basic_subm_creators_dict=False, additional_init_logic=deepcv_module._additional_init_logic)
    elif subm_type == yaml_tokens.NAS_LAYER_CHOICE:
        deepcv_module._uses_nni_nas_mutables = True
        if not allow_mutable_layer_choices:
            raise ValueError(f'Error: nested LayerChoices are forbiden, cant specify a NNI NAS Mutable LayerChoice as a candidate of another LayerChoice ("{subm_type}").')
        # List of alternative submodules: nni_mutables.LayerChoice + makes sure candidate submodules names can't be referenced : LayerChoice candidates may have names (OrderedDict instead of List) but references are only allowed on 'yaml_tokens.NAS_LAYER_CHOICE' global name)
        # for more details on `LayerChoice`, see https://nni.readthedocs.io/en/latest/NAS/NasReference.html#nni.nas.pytorch.mutables.LayerChoice
        if not isinstance(params, Dict[str, Any]) or yaml_tokens.NAS_LAYER_CHOICE_CANDIDATES not in params or any([p not in {yaml_tokens.NAS_LAYER_CHOICE_CANDIDATES, yaml_tokens.NAS_LAYER_REDUCTION_FN, yaml_tokens.NAS_MUTABLE_RETURN_MASK} for p in params.keys()]):
            raise ValueError(f'Error: Parameters of a "{yaml_tokens.NAS_LAYER_CHOICE}" submodule specification must be a Dict which at least contains a `_candidates` parameter. '
                             f'(And may eventually specify `{yaml_tokens.NAS_LAYER_REDUCTION_FN}`, `{yaml_tokens.NAS_MUTABLE_RETURN_MASK}` and/or `{yaml_tokens.SUBMODULE_NAME}` parameter(s)). NNI Mutable LayerChoice submodule params received: "{params}"')
        prefix = f'{default_submodule_prefix}{submodule_pos}_candidate_'
        reduction = getattr(params, yaml_tokens.NAS_LAYER_REDUCTION_FN, DEFAULT_LAYER_CHOICE_REDUCTION)
        return_mask = getattr(params, yaml_tokens.NAS_MUTABLE_RETURN_MASK, False)

        # Parse candidates/alternative submodules from params (recursive call to `_parse_torch_module_from_submodule_spec`)
        candidate_refs, candidates = list(), OrderedDict()
        for j, candidate_spec in enumerate(params[yaml_tokens.NAS_LAYER_CHOICE_CANDIDATES]):
            candidate_name, candidate = _parse_torch_module_from_submodule_spec(deepcv_module, submodule_spec=candidate_spec, submodule_pos=j, subm_creators=subm_creators,
                                                                                default_submodule_prefix=prefix, allow_mutable_layer_choices=False)
            if getattr(candidate, 'referenced_submodules', None) is not None:
                candidate_refs.extend(candidate.referenced_submodules)
            else:
                # Ignore any tensor references if candidate doesn't need tensor references (no `referenced_submodules` attribute so subm is assumed to not take `referenced_submodules_out` argument)
                # NOTE: We assume that `referenced_submodules` attribute is reserved to modules which takes `referenced_submodules_out` arg (probably ForwardCallbackSubmodule)
                def _forward_monkey_patch(*args, referenced_submodules_out: Dict[str, TENSOR_OR_SEQ_OF_TENSORS_T] = None, **kwargs):
                    return candidate.forward(*args, **kwargs)
                candidate.forward = _forward_monkey_patch
            candidates[candidate_name] = candidate

        # Instanciate NNI NAS Mutable LayerChoice from parsed candidates and parameters
        module = nni_mutables.LayerChoice(op_candidates=candidates, reduction=reduction, return_mask=return_mask, key=subm_name)
        # Candidates tensor references are agregated and stored in parent LayerChoice so that `deepcv_module._submodule_references` only stores references of top-level submodules (not nested candidates)
        module.referenced_submodules = candidate_refs
    else:
        # Parses a regular NN submodule from specs. (either based on a submodule creator or directly a `torch.nn.Module` type or string identifier)
        if isinstance(subm_type, str):
            # Try to find sub-module creator or a torch.nn.Module's `__init__` function which matches `subm_type` identifier
            fn_or_type = subm_creators.get(subm_type)
            if not fn_or_type:
                # If we can't find suitable function in module_creators, we try to evaluate function name (allows external functions to be used to define model's modules)
                try:
                    fn_or_type = deepcv.utils.get_by_identifier(subm_type)
                except Exception as e:
                    raise RuntimeError(f'Error: Could not locate module/function named "{subm_type}" given module creators: "{subm_creators.keys()}"') from e
        else:
            # Specified submodule is assumed to be directly a `torch.nn.Module` or `Callable[..., torch.nn.Module]` type which will be instanciated with its respective parameters as possible arguments according to its `__init__` signature (`params` and global NN spec. parameters)
            fn_or_type = subm_type

        # Create layer/block submodule from its module_creator or its `torch.nn.Module.__init__()` method (`fn_or_type`)
        submodule_signature_params = inspect.signature(fn_or_type).parameters
        params_with_globals['prev_shapes'] = deepcv_module._features_shapes  # Make possible to take a specific argument to get previous submodules output tensor shapes
        params_with_globals['input_shape'] = deepcv_module._features_shapes[-1]  # Make possible to take a specific argument to get input tensor shape(s) from previous submodule
        params_with_globals['input_shapes'] = deepcv_module._features_shapes[-1]  # Make possible to take a specific argument to get input tensor shape(s) from previous submodule
        provided_params = {n: v for n, v in params_with_globals.items() if n in submodule_signature_params}
        # Add `submodule_params` and ``prev_shapes` to `provided_params` if they are taken by submodule creator (or `torch.nn.Module` constructor)
        if 'submodule_params' in submodule_signature_params:
            # `submdule_params` parameter wont provide a param which is already provided through `provided_params` (either provided through `submdule_params` dict or directly as an argument named after this parameter `n`)
            provided_params['submodule_params'] = {n: v for n, v in params.items() if n not in provided_params}
        # Create submodule from its submdule creator or `torch.nn.Module` constructor
        module = fn_or_type(**provided_params)

        # Process submodule creators output `torch.nn.Module` so that `deepcv.meta.submodule_creators.ForwardCallbackSubmodule` submodules instances are handled in a specific way for output tensor references (e.g. dense/residual) and  NNI NAS Mutable InputChoice support. (these modules are defined by forward pass callbacks which may be fed with referenced sub-module(s) output and to previous sub-module output)
        if isinstance(module, ForwardCallbackSubmodule):
            # Submodules which are instances of `deepcv.meta.submodule_creators.ForwardCallbackSubmodule` are handled sperately allowing output tensor (residual/dense) references and NNI NAS Mutable InputChoice support. (`DeepcvModule`-specific `torch.nn.Module` defined from a callback called on forward passes)
            _setup_forward_callback_submodule(deepcv_module, subm_name, submodule_params=params, forward_callback_module=module)
        elif not isinstance(module, torch.nn.Module):
            raise RuntimeError(f'Error: Invalid sub-module creator function or type: '
                               f'Must either be a `torch.nn.Module` (Type or string identifier of a Type) or a submodule creator which returns a `torch.nn.Module`.')
    return subm_name, module


def _subm_name_and_params_from_spec(submodule_spec: Union[Dict, Type[torch.nn.Module], str, Callable[..., torch.nn.Module]], default_subm_name: str, existing_subm_names: Sequence[str]) -> Tuple[str, Dict, Union[Type[torch.nn.Module], str, Callable[..., torch.nn.Module]]]:
    # Retreive submodule parameters and type for all possible submodule spec. senarios
    subm_type, params = list(submodule_spec.items())[0] if isinstance(submodule_spec, Dict) else (submodule_spec, {})
    if isinstance(params, List) or isinstance(params, Tuple):
        # Architecture definition specifies a sub-module name explicitly
        subm_name, params = params[0], params[1]
    elif isinstance(params, str):
        # Architecture definition specifies a sub-module name explicitly without any other sub-module parameters
        subm_name, params = params, dict()
    elif isinstance(params, Dict[str, Any]) and yaml_tokens.SUBMODULE_NAME in params:
        # Allow to specify submodule name in submodule parameters dict instead of throught Tuple/List usage (`yaml_tokens.SUBMODULE_NAME` parameter)
        subm_name = params[yaml_tokens.SUBMODULE_NAME]
        del params[yaml_tokens.SUBMODULE_NAME]

    # Checks whether if subm_name is invalid or duplicated
    if subm_name in existing_subm_names or subm_name == r'' or not isinstance(subm_name, str):
        raise ValueError(f'Error: Invalid or duplicate sub-module name/label: "{subm_name}"')
    # Checks if `params` is a valid
    if not isinstance(params, Dict):
        raise RuntimeError(f'Error: Architecture sub-module spec. must either be a parameters Dict, or a submodule name along with parameters Dict, but got: "{params}".')

    return (subm_name, params, subm_type)


def _setup_forward_callback_submodule(deepcv_module: 'deepcv.meta.base_module.DeepcvModule', subm_name: str, submodule_params: Dict[str, Any], forward_callback_module: 'deepcv.submodule_creators.ForwardCallbackSubmodule') -> Tuple[str, Optional[torch.nn.Module]]:
    """ Specfic model definition logic for submodules based on forward pass callbacks (`deepcv.meta.submodule_creators.ForwardCallbackSubmodule` submodule instances are handled sperately allowing output tensor (residual/dense) references and NNI NAS Mutable InputChoice support).
    Allows referencing other submodule(s) output tensor(s) (`yaml_tokens.FROM` usage) and NNI NAS Mutable InputChoice (`yaml_tokens.FROM_NAS_INPUT_CHOICE` usage).
    """
    deepcv_module._uses_forward_callback_submodules = True
    # yaml_tokens.FROM_NAS_INPUT_CHOICE occurences in `submodule_params` are handled like yaml_tokens.FROM entries: nni_mutables.InputChoice(references) + optional parameters 'n_chosen' (None by default, should be an integer between 1 and number of candidates)
    if yaml_tokens.FROM_NAS_INPUT_CHOICE in submodule_params:
        deepcv_module._uses_nni_nas_mutables = True
        n_chosen = submodule_params['n_chosen'] if yaml_tokens.FROM_NAS_INPUT_N_CHOSEN in submodule_params else None
        n_candidates = len(submodule_params[yaml_tokens.FROM_NAS_INPUT_CHOICE])
        mask = submodule_params[yaml_tokens.NAS_MUTABLE_RETURN_MASK] if yaml_tokens.NAS_MUTABLE_RETURN_MASK in submodule_params else False
        forward_callback_module.mutable_input_choice = nni_mutables.InputChoice(n_candidates=n_candidates, n_chosen=n_chosen,
                                                                                return_mask=mask, key=subm_name, reduction='none')

        if yaml_tokens.FROM in submodule_params:
            raise ValueError(f'Error: Cant both specify "{yaml_tokens.FROM}" and "{yaml_tokens.FROM_NAS_INPUT_CHOICE}" in the same submodule '
                             '(You should either choose to use NNI NAS Mutable InputChoice candidate reference(s) or regular tensor reference(s)).')
    elif yaml_tokens.NAS_MUTABLE_RETURN_MASK in submodule_params or yaml_tokens.FROM_NAS_INPUT_N_CHOSEN:
        raise ValueError(f'Error: Cannot specify "{yaml_tokens.NAS_MUTABLE_RETURN_MASK}" nor "{yaml_tokens.FROM_NAS_INPUT_N_CHOSEN}" without using "{yaml_tokens.FROM_NAS_INPUT_CHOICE}".'
                         f'("{yaml_tokens.NAS_MUTABLE_RETURN_MASK}" and "{yaml_tokens.FROM_NAS_INPUT_N_CHOSEN}" is an optional parameter reserved for NNI NAS InputChoice usage).')

    # Store any sub-module name/label references (used to store referenced submodule's output features during model's forward pass in order to reuse these features later in a forward callback (e.g. for residual links))
    if yaml_tokens.FROM in submodule_params or yaml_tokens.FROM_NAS_INPUT_CHOICE in submodule_params:
        # Allow multiple referenced sub-module(s) (`yaml_tokens.FROM` entry can either be a list/tuple of referenced sub-modules name/label or a single sub-module name/label)
        tensor_references = submodule_params[yaml_tokens.FROM] if yaml_tokens.FROM in submodule_params else submodule_params[yaml_tokens.FROM_NAS_INPUT_CHOICE]
        forward_callback_module.referenced_submodules = [tensor_references, ] if isinstance(tensor_references, str) else list(tensor_references)

#_________________________________________________ NN SPEC UNIT TESTS _________________________________________________#


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
