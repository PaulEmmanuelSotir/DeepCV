#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" NNI integration for DeepCV meta module - nni_tools.py - `DeepCV`__
NNI integration for DeepCV with various tooling for easier/unified usage of NNI HP and NNI NAS (Single-Shot NAS and Classic NAS) APIs.
.. moduleauthor:: Paul-Emmanuel Sotir

# To-Do List
    - TODO: Make usage of nnictl tensorboard during HP searches? (may better handles tensorboard logs along with cleanner tb server starting and stoping)
"""
import os
import json
import types
import logging
import functools
import subprocess
import multiprocessing
from pathlib import Path
from typing import Sequence, Iterable, Callable, Dict, Tuple, Any, Union, Optional, List, Type, Set

import mlflow
import anyconfig

import torch
import torch.nn
from torch.utils.data import DataLoader, Dataset

import nni
import nni.nas.pytorch.trainer
import nni.nas.pytorch.mutator
import nni.nas.pytorch.mutables
import nni.nas.pytorch.classic_nas
import nni.nas.pytorch.base_mutator
from nni.env_vars import trial_env_vars
# TODO: add darts, pdarts, cdarts and eventually textnas once NNI removed its dependency on Apex, which is unescessary since PyTorch now integrates it (builtin)
from nni.nas.pytorch import enas, spos, proxylessnas

import deepcv.utils
from deepcv.utils import NL
from . import hyperparams
from .data import datasets
from .types_aliases import *

__all__ = ['NNI_SINGLE_SHOT_NAS_ALGORITHMS', 'NNI_CLASSIC_NAS_ALGORITHMS', 'NNI_HP_TUNERS', 'NNI_SINGLE_SHOT_NAS_MUTATORS' 
           'is_nni_gen_search_space_mode', 'is_nni_run_standalone', 'get_nni_or_mlflow_experiment_and_trial', 'model_contains_nni_nas_mutable',
           'gen_classic_nas_search_space', 'nni_single_shot_neural_architecture_search', 'handle_nni_nas_trial', 'run_nni_hp_search', 'gen_nni_config', 'sample_nni_hp_space',
           'hp_search', 'get_hp_position_in_search_space', 'generate_hp_space_template']
__author__ = 'Paul-Emmanuel Sotir'

#______________________________________________ NNI INTEGRATION CONSTANTS _____________________________________________#


# TODO: Add Darts, CDarts and PDarts once NNI removed its dependency on Apex (Apex is part of PyTorch and is an unescessary dependency):
# 'DARTS': darts.DartsTrainer, 'P-DARTS': pdarts.PdartsTrainer, 'CDARTS': cdarts.CdartsTrainer + eventually add: 'TestNAS': textnas.TextNasTrainer}
""" NNI Single-Shot NAS algorithms, each associated to its respective Trainer (See also there respective mutators in NNI documentation) """
NNI_SINGLE_SHOT_NAS_ALGORITHMS = {'ENAS': enas.EnasTrainer, 'SPOS': spos.SPOSSupernetTrainer,
                                  'ProxylessNAS': proxylessnas.ProxylessNasTrainer}
NNI_CLASSIC_NAS_ALGORITHMS = {'PPOTuner', 'RandomTuner'}  # NNI tuners which supports search spaces generated from NNI NAS Mutable API (NNI Classic/Distributed NAS)
NNI_HP_TUNERS = {}  # TODO: fill this with NNI HP tuners
NNI_SIGNLE_SHOT_NAS_MUTATORS = ...  # TODO: More support for NNI NAS Mutators

#_____________________________________________ NNI INTEGRATION FUNCTIONS ______________________________________________#


# TODO: remove it
# def nni_hp_with_singleshot_nas_trial(params) -> bool:
#     """ Returns a boolean indicating whether we are performing an NNI HP search with NNI Single-Shot NAS training as trial code  (Nested usage of NNI HP and NNI SingleShot NAS APIs) """
#     return trial_env_vars.NNI_PLATFORM == ... and asked_for_nas and single_shot_nas_algorithm


def is_nni_gen_search_space_mode() -> bool:
    # If `NNI_GEN_SEARCH_SPACE` is defined, then NNI is in dry run mode for NAS search space generation
    return 'NNI_GEN_SEARCH_SPACE' in os.environ


def is_nni_run_standalone() -> bool:
    """ Simple helper function which returns whether NNI is in standalone trial run mode """
    return nni.get_experiment_id() == r'STANDALONE' and nni.get_trial_id() == r'STANDALONE' and nni.get_sequence_id() == 0
    # TODO: Standalone mode can also be detected with this condition (and make sure it is relevant to return 'True' on NNI unit tests (from constructor code of NNI NAS 'ClassicMutator' https://nni.readthedocs.io/en/latest/_modules/nni/nas/pytorch/classic_nas/mutator.html#ClassicMutator))
    # return trial_env_vars.NNI_PLATFORM is None or trial_env_vars.NNI_PLATFORM == "unittest"


def get_nni_or_mlflow_experiment_and_trial() -> Tuple[Optional[str], Optional[str]]:
    """ Helper function which returns NNI experiment name and trial ID if NNI isn't in Standalone mode or, otherwise, returns MLFlow experiment name and run ID if there is an active MLFlow run. 
    Returns (None, None) if NNI is in standalone mode and there is no active MLFLow run.
    """
    if is_nni_run_standalone():
        exp, run = deepcv.utils.mlflow_get_experiment_run_info()
        return (None, None) if exp is None else (exp.name, str(run.run_id))
    return (nni.get_experiment_id(), nni.get_trial_id())


def model_contains_nni_nas_mutable(model: torch.nn.Module):
    """ Look for any NNI NAS Mutable `InputChoice` or `LayerChoice` among all child module using PyTorch `modules` recursive method
    Returns `True` if given model contains any NNI NAS Mutable (`InputChoice` or `LayerChoice`), returns `False` otherwise.
    """
    def _is_nni_nas_mutable(child_module: torch.nn.Module):
        if getattr(child_module, 'uses_nni_nas_mutables', None):
            return child_module.uses_nni_nas_mutables(recursive=False)  # No need to recurse here as we are already iterating over all child modules recursively
        else:
            return isinstance(child_module, (nni.nas.pytorch.mutables.LayerChoice, nni.nas.pytorch.mutables.InputChoice))
    return any(map(model.modules(), _is_nni_nas_mutable))


# def nni_classic_neural_architecture_search_trial(model: torch.nn.Module, training_procedure: Callable[[Dict[str, Dataset], torch.nn.Module, HYPERPARAMS_T], Any], *other_training_args, **other_training_kwargs):
#     """ Applies choices among NNI NAS mutable layers and inputs to model architecture and train it using provided training procedure.
#     NNI Classic NAS algorithm (either PPO Tuner or Random Tuner) samples a fixed architeture from mutable NN architecture search space.
#     NOTE: NNI NAS also supports various single shot neural architecture search algorithms for faster architecture optimization (performed in a single trial instead of one trial per evaluated/possible architecture).
#     If this is the first call to NNI Classic NAS API, NNI's `get_and_apply_next_architecture` will first generate JSON search space based on given `model` (NNI dry run mode, for more details, see https://nni.readthedocs.io/en/latest/NAS/NasReference.html#nni.nas.pytorch.classic_nas.get_and_apply_next_architecture)
#     Args:
#         - model: Model NAS search space to be sampled from and trained. Provided model should be a regular PyTorch module which contains at least one NNI mutable layer(s) and/or mutable input(s).
#         - training_procedure: Training procedure of your own used to run a trial. This function should take model to train as its first argument (sampled NN architecture from `model`).
#         - *other_training_args: Any positional arguments which should be provided to `training_procedure`, except for the model to train which is already provided by NNI Architecture sampling.
#         - **other_training_kwargs: Any other keyword arguments to be provided to `training_procedure`.
#     Returns the training results returned by given `training_procedure`. Note that `training_procedure` should at least return a dict-like object with a `best_valid_loss` entry which will be reported to NNI NAS API (float scalar indicating sampled architecture performances).
#     """
#     nni.nas.pytorch.classic_nas.get_and_apply_next_architecture(model)
#     results = training_procedure(model, *other_training_args, **other_training_kwargs)
#     nni.report_final_result(results['best_valid_loss'])
#     return results


def gen_classic_nas_search_space(architecture_search_space_path: Union[str, Path], trial_cmd: str, trial_code_dir: Union[Path, str, None] = None) -> bool:
    # `--trial_dir` defaults to './' when not provided
    optional_code_dir_arg = list() if trial_code_dir is None else ['--trial_dir', str(trial_code_dir)]
    sub = subprocess.run(['nnictl', 'ss_gen', '--trial_command', trial_cmd, *optional_code_dir_arg, '--file', str(architecture_search_space_path)],
                         capture_output=True, universal_newlines=True, check=False)
    if sub.returncode != os.EX_OK:
        logging.info(f'Sucessfull JSON NNI NAS search space generation using `nnictl ss_gen` command. STDOut from `nnictl`: "{sub.stdout}"')
    else:
        logging.error(f'Failed to generate JSON NNI NAS search space using `nnictl ss_gen` command. STDErr from `nnictl`: "{sub.stderr}"')
    return sub.returncode == os.EX_OK

""" Default accepted batch mapping target(s) and input(s) names (see `deepcv.meta.ignite_training.get_inputs_and_targets_from_batch`) """
DEFAULT_ACCEPTED_BATCH_MAPPING_TARGET_KEYS = set({'target', 'y', 'label', 'ground_truth', 'groundtruth'}) # The same keys with a trailing 's' are also accepted/valid target keys
DEFAULT_ACCEPTED_BATCH_MAPPING_TARGET_KEYS += {f'{key}s' for key in DEFAULT_ACCEPTED_BATCH_MAPPING_TARGET_KEYS}
DEFAULT_ACCEPTED_BATCH_MAPPING_INPUT_KEYS = set({'input', 'in', 'x', 'image'})  # The same keys with a trailing 's' are also accepted/valid input keys
DEFAULT_ACCEPTED_BATCH_MAPPING_INPUT_KEYS += {f'{key}s' for key in DEFAULT_ACCEPTED_BATCH_MAPPING_TARGET_KEYS}

def get_inputs_and_targets_from_batch(batch: Union[TENSOR_OR_SEQ_OF_TENSORS_T, Sequence[TENSOR_OR_SEQ_OF_TENSORS_T], Mapping[TENSOR_OR_SEQ_OF_TENSORS_T]], device: Union[str, torch.device] = None, convert_mapping_to_list: bool = False, mapping_input_keys: Set[str] = DEFAULT_ACCEPTED_BATCH_MAPPING_INPUT_KEYS, mapping_targets_keys: Set[str] = None):
    """ Returns inputs and targets obtained from `batch`.  
    Input tensor(s) can be provided in various ways: `batch` can either be a single input tensor, a sequence of `TENSOR_OR_SEQ_OF_TENSORS_T` or a mapping of `TENSOR_OR_SEQ_OF_TENSORS_T`.  
    If obtained input(s) and/or target(s) are lists (not maps/dicts) and only have one element, then those are `squeezed` so that their only entry is returned instead. 

    Args:  
        - batch: Input batch from which returned inputs and targets are obtained  
        - device: If not `None`, input and target tensor(s) are moved to provided device  
        - convert_mapping_to_list: If `True`, then, when input `batch` is a mapping/dict of str keys associated to `TENSOR_OR_SEQ_OF_TENSORS_T`, returned input(s) and target(s) are lists of tensors instead of dicts.  
        - mapping_input_keys: Used for mapping/dict input `batch`: Input tensor(s) are choosen from `batch` mapping according to `mapping_input_keys` keys (returned inputs will only contain tensors from `batch` which have its name specified in `mapping_input_keys`)  
        - mapping_targets_keys: Used for mapping/dict input `batch`: If `None` (default), all non-input tensors from `batch` mapping are considered as targets. Otherwise, if this argument is provided (e.g. set to `deepcv.meta.ignite_training.DEFAULT_ACCEPTED_BATCH_MAPPING_TARGET_KEYS`), returned targets only contains tensors from `batch` which are named after keys in `mapping_targets_keys`.  
    
    NOTE: Logs a warning if there are common keys between `mapping_input_keys` and `mapping_targets_keys`  
    NOTE: When `mapping_targets_keys` isn't `None` and when `batch` is a mapping/dict, if there are some 'unkown' tensor(s) keys in `batch` which are not present in `mapping_input_keys` nor in `mapping_targets_keys`, then a warning is logged.  
    """
    is_mapping = isinstance(batch, Mapping)
    if is_mapping:
        inputs = {n: x for n, x in batch.items() if n in mapping_input_keys}
        if mapping_targets_keys is None:
            targets = {n: x for n, x in batch.items() if n not in mapping_input_keys}
        else:
            targets = {n: x for n, x in batch.items() if n in mapping_targets_keys}
            overlaping_keys = [k for k in mapping_targets_keys if k in mapping_input_keys]
            if len(overlaping_keys) > 0:
                logging.warn(f'Warning: there are common/overlaping keys ("{overlaping_keys}") betwen target(s) and input(s) accepted keys:'
                             f'"mapping_input_keys={mapping_input_keys}", "mapping_targets_keys={mapping_targets_keys}"')
            if len(targets) + len(inputs) != len(batch):
                logging.warn('Warning: Some entries from `batch` input tensor(s) map have been ignored because those have unrecognized key/name (not in `mapping_targets_keys` nor in `mapping_input_keys`)')
        if convert_mapping_to_list:
            inputs, targets = list(inputs.values()), list(targets.values())
    else:
        if isinstance(batch, Sequence):
            if len(batch) == 0:
                inputs, targets = list(), list()
            else:
                inputs, *targets = list(batch)
                targets = [t[0] if len(t) == 1 else t for t in targets]
        else:
            inputs, targets = batch, list()

        if len(targets) == 1:
            targets = targets[0]
        if len(inputs) == 1:
            inputs = inputs[0]

    if device not in (None, ''):
        inputs = ignite.utils.convert_tensor(inputs, device=device, non_blocking=True)
        if is_mapping and not convert_mapping_to_list:
            targets = {n: ignite.utils.convert_tensor(x, device=device, non_blocking=True) for n, x in targets.items()}
        else:
            targets = [ignite.utils.convert_tensor(x, device=device, non_blocking=True) for x in targets]
    return inputs, targets


def _single_shot_nas_retrain_for_eval(model: torch.nn.Module, criterion: LOSS_FN_T, metrics: Dict[str, METRIC_FN_T], trainset: DataLoader, max_iters: int, module_types_to_be_retrained: Tuple[Type[torch.nn.Module]]) -> Optinal[AverageMeterGroup]:
    """ Function re-training model for evaluation/metric-reporting which is nescessary when model contains batch normmalization or any other techniques leveraging any 'learned' statistics which could be biased by single-shot NAS algorithm.  
    For example, Single-Shot NAS SPOS algorithm trains a 'superset' network which could have different batch norm statistics than statistics needed for unbiased evaluation of fixed architecture sampled from this 'superset' model architecture search space.  
    .. See NNI NAS example from which is inspired this code: [tester.py in SPOS NNI Single-Shot NAS example](https://github.com/microsoft/nni/blob/master/examples/nas/spos/)  
    TODO: make sure this is needed for any SingleShot NAS algorithm or anly perform this when using SPOS.  
    """    
    needs_to_be_retrained = False

    with torch.no_grad():
        for submodule in model.modules():
            if isinstance(submodule, module_types_to_be_retrained):
                needs_to_be_retrained = True
                submodule.running_mean = torch.zeros_like(submodule.running_mean)
                submodule.running_var = torch.ones_like(submodule.running_var)

        if needs_to_be_retrained:
            logger.info('Performing a re-scaling/reset of model batch statistics for unbasied evaluation of architecture during Single-Shot Neural Architecture Search... (e.g. BatchNorm statistics needs to be reset/retrained)')
            model.train()
            meters = AverageMeterGroup()
            for step, batch in deepcv.utils.progress_bar(zip(range(max_iters), trainset), desc='Retraining model before NNI Single-Shot NAS evaluation:'):
                inputs, targets = get_inputs_and_targets_from_batch(batch, device=device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                batch_metrics = {'batch_loss': loss.item(), **{metric_fn(logits, targets).item() for metric_name, metric_fn in metrics.items()}}
                meters.update(metrics)
            return meters

def nni_single_shot_eval(model: torch.nn.Module, criterion: LOSS_FN_T, metrics: Dict[str, METRIC_FN_T], trainset: DataLoader, max_iters: int = 10000, module_types_to_be_retrained: Tuple[Type[torch.nn.Module]] = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)) -> METRICS_DICT_T:
    _single_shot_nas_retrain_for_eval(model, criterion, metrics, trainset, max_iters, module_types_to_be_retrained)
    

    if final_eval:
        nni.report_final_result(metrics)
    else:
        nni.report_intermediate_result(metrics)
    return metrics

#TODO: refactor loss usage like in ingite_training and use `get_inputs_and_targets_from_batch` in dataloader ouput transform (harmonize training procedures in `ingite_training.train`, `_single_shot_nas_retrain_for_eval` and `nni_single_shot_eval`)
def nni_single_shot_neural_architecture_search(hp: HYPERPARAMS_T, model: torch.nn.Module, losses: LOSS_FN_TERMS_T, datasets: Tuple[Dataset], opt: Type[torch.optim.Optimizer], backend_conf: 'ignite_training.BackendConfig' = None,
                                               loss_weights: LOSS_TERMS_WEIGHTS_T = None, metrics: Dict[str, METRIC_FN_T] = {}, callbacks_handler: deepcv.utils.EventsHandler = None, final_architecture_path: Union[str, Path] = None) -> Tuple[METRICS_DICT_T, Optional[Path], str]:
    """ Train model with provided NAS trainer in order to find out the best NN architecture by training a superset NN instead of performing multiple trainings/trials for each/many possible architectures.  
    Args:
        - hp: Hyperparameter dict, see ```deepcv.meta.ignite_training._check_params`` to see required and default training (hyper)parameters  
        - model: Pytorch ``torch.nn.Module`` to train  
        - losses: Loss(es) module(s) and/or callables to be used as training criterion(s) (may be a single loss function/module, a sequence of loss functions or a mapping of loss function(s) assiciated to their respective loss name(s)).  
            .. See `loss_weights` argument for more details on multiple loss terms usage and weighting.
        - datasets: Tuple of pytorch Dataset giving access to trainset, validset and an eventual testset  
        - opt: Optimizer type to be used for gradient descent  
        - backend_conf: Backend information defining distributed configuration (available GPUs, whether if CPU or GPU are used, distributed node count, ...), see ``deepcv.meta.ignite_training.BackendConfig`` class for more details.  
        - loss_weights: Optional weight/scaling/importance vector or sequence to be applied to each loss terms (defaults to 1. when `None`). This argument should contain as many scalars as there are loss terms/functions in `losses` argument. All weight values are casted to `torch.float32` and L1-norm (sum) should be different from zero due to the mean operator (loss terms ponderated sum is diveded by L1-norm of weights).  
            NOTE: Each scalar in `loss_weights` weights its respective loss term so that the total loss on which model is trained is the ponderated mean of each loss terms (e.g. when `loss_weights` contains `1.` values for each loss term, then the final loss is the mean of each of those. Another example: if `loss_weights` only contains `len()` values, then the loss on which model is trained is the sum of each terms)  
            NOTE: You may provide a mapping of weights instead of a Sequence in case you need to apply weights/factors to their respective term identified by their names (`losses` should also be a mapping in this case)
        - metrics: Additional metrics dictionnary (loss is already included in metrics to be evaluated by default)  
        - callbacks_handler: Callbacks Events handler. If not `None`, events listed in `deepcv.meta.ignite_training.TRAINING_EVENTS` will be fired at various steps of training process, allowing to extend `deepcv.meta.ignite_training.train` functionnalities ("two-way" callbacks ).  
        - final_architecture_path: File path where the final/optimal fixed architecture found by Single-Shot NAS algorithm have to be exported (JSON file which contains NNI NAS Mutable choices needed to obtain the fixed architecture from the model search space).  

    Returns a pathlib.Path to a JSON file storing the best NN model architecture found by NNI Single-Shot NAS (JSON file storing mutable layer(s)/input(s) choices made in model search space in order to define best fixed architeture found; This file is also logged to mlflow if there is an active run).  
    NOTE: Support for SPOS SingleShot NAS is untested and may be partial for now, see [NNI NAS SPOS documentation](https://nni.readthedocs.io/en/latest/NAS/SPOS.html) for more details on Evolution Tuner usage to find best model architecture.  

    *To-Do List*
        - # TODO: convert ignite metrics for NNI NAS trainer usage if needed (to Callable[['outout', 'target'], Dict[str, float]])
        - # TODO: reuse code from ignite training for output path and log final architecture as mlflow artifact
        - # TODO: Allow resuming an NNI single shot NAS experiment throught 'hp['resume_from']' parameter (if possible easyly using NNI API?)
        - # TODO: Add support for two-way callbacks using deepcv.utils.EventsHandler in a similar way than ignite_training.train (once ignite_training fully support it)
    """
    from .ignite_training import BackendConfig
    if backend_conf is None:
        backend_conf = BackendConfig()
    experiment_name, run_id = get_nni_or_mlflow_experiment_and_trial()
    run_info_msg = f'(Experiment: "{experiment_name}", run_id: "{run_id}")'
    logging.info(f'Starting Single-Shot Neural Architecture Search (NNI NAS API) training over NN architecture search space {run_info_msg}.')

    TRAINING_HP_DEFAULTS = {'optimizer_opts': ..., 'epochs': ..., 'batch_size': None, 'nni_single_shot_nas_algorithm': ..., 'output_path': Path.cwd() / 'data' / '04_training', 'log_output_dir_to_mlflow': True,
                            'log_progress_every_iters': 100, 'seed': None, 'resume_from': '', 'deterministic_cudnn': False, 'nas_mutator': None, 'nas_mutator_kwarg': dict(), 'nas_trainer_kwargs': dict()}
    hp, _ = hyperparams.to_hyperparameters(hp, TRAINING_HP_DEFAULTS, raise_if_missing=True)
    deepcv.utils.setup_cudnn(deterministic=hp['deterministic_cudnn'], seed=backend_conf.rank + hp['seed'])  # In distributed setup, we need to have different seed for each workers
    model = model.to(backend_conf.device, non_blocking=True)
    loss = loss.to(backend_conf.device) if isinstance(loss, torch.nn.Module) else loss
    num_workers = max(1, (backend_conf.ncpu - 1) // (backend_conf.ngpus_current_node if backend_conf.ngpus_current_node > 0 and backend_conf.distributed else 1))
    optimizer = opt(model.parameters(), **hp['optimizer_opts'])
    trainset, *validset_testset = datasets
    output_path = ingite_training.add_training_output_dir(hp['output_path'], backend_conf, prefix='single_shot_nas_')
    # TODO: use this output_path in trainer and export final architecture to this directory too
    # TODO: use ingite_training function to setup distributed training?

    # Creates HP scheduler from hp if respective hp arguments have been provided by user
    scheduler = None
    if hp['scheduler'] is not None:
        args_to_eval = hp['scheduler']['eval_args'] if 'eval_args' in hp['scheduler'] else {}
        scheduler_kwargs = {n: eval(v, {'hp': hp, 'iterations': len(trainset)}) if n in args_to_eval else v
                            for n, v in hp['scheduler']['kwargs'].items()}
        scheduler = nni.nas.pytorch.callbacks.LRSchedulerCallback(scheduler=hp['scheduler']['type'](optimizer=optimizer, **scheduler_kwargs))
    nas_trainer_callbacks = [scheduler, ]  # TODO: ... add user provided callbacks

    if not is_nni_run_standalone() and not is_nni_gen_search_space_mode():
        class _ReportToNNICallback(nni.nas.pytorch.callbacks.Callback):
            def __init__(self, epochs=hp['epochs']):
                self.epochs = epochs

            def on_epoch_end(self, epoch):
                # TODO: find a way to retreive metrics or evaluate model on my own (meters = AverageMeterGroup() ...), see https://nni.readthedocs.io/en/latest/_modules/nni/nas/pytorch/enas/trainer.html
                meters = ...
                if epoch >= self.epochs:
                    nni.report_final_result(meters)
                else:
                    nni.report_intermediate_result(meters)
        nas_trainer_callbacks.append(_ReportToNNICallback(hp))

    nas_trainer_kwargs = hp['nas_trainer_kwargs']
    nas_mutator = hp['nas_mutator']
    if nas_mutator is not None:
        if isinstance(nas_mutator, str):
            nas_mutator = deepcv.utils.get_by_identifier(nas_mutator)
        if not issubclass(nas_mutator, nni.nas.pytorch.base_mutator.BaseMutator):
            raise TypeError('Error: NNI SingleShot NAS Mutator argument "nas_mutator" must either be a "nni.nas.pytorch.mutables.Mutator" Type or a string identifier which resolves to a "nni.nas.pytorch.mutables.Mutator" Type.')
        nas_trainer_kwargs['mutator'] = nas_mutator(**hp['nas_mutator_kwarg'])  # Instanciate user-provided mutator
    if hp['batch_size'] is not None:
        nas_trainer_kwargs['batch_size'] = hp['batch_size']

    train_type = NNI_SINGLE_SHOT_NAS_ALGORITHMS[hp['nni_single_shot_nas_algorithm']]
    trainer = train_type(model=model, loss=loss, metrics=metrics, optimizer=optimizer, num_epochs=hp['epochs'],
                         trainset=trainset, validset=validset_testset[0], num_workers=num_workers, device=backend_conf.device,
                         log_frequency=hp['log_progress_every_iters'], callbacks=nas_trainer_callbacks, **nas_trainer_kwargs)

    # Train model with provided NAS trainer in order to find out the best NN architecture by training a superset NN instead of performing multiple trainings/trials for each/many possible architectures
    trainer.train()
    logging.info(f'Single-shot NAS training done. Validating model architecture... {run_info_msg}')
    trainer.validate()
    logging.info(f'Saving obtained NN architecture from NNI Single-Shot NAS algorithm as a JSON file and logging it to mlfow if possible... {run_info_msg}')

    # Print resulting architecture as a JSON string and save it to a JSON file if `final_architecture_path` isn't `None` (and is valid)
    architecture_choices = trainer.mutator.export()
    json_choices = deepcv.utils.replace_newlines(json.dumps(architecture_choices, indent=2, sort_keys=True, cls=nni.nas.pytorch.trainer.TorchTensorEncoder))
    logging.info(f'Final/best NN architeture obtained from NNI Single-Shot NAS wont be saved to a JSON file as `final_architecture_path` is `None`. {run_info_msg}{NL}'
                 f'NAS Mutable choices:{NL}'
                 f'``` json{NL}{json_choices}{NL}```')

    logging.info(f'Saving final/best NN architeture obtained from NNI Single-Shot NAS (and may log it to MLFLow artifacts). {run_info_msg}')

    if final_architecture_path is not None:
        final_architecture_path = Path(final_architecture_path)
        with final_architecture_path.open(mode='w', newline=NL) as json_file:  # 'w' mode will replace any existing file
            json_file.write(json_choices)  # export the final architecture to a JSON file
    if mlflow.active_run() is not None:
        if final_architecture_path is not None:
            mlflow.log_artifact(str(final_architecture_path))
            mlflow.set_tag('final_single_shot_nas_architecture_path', str(final_architecture_path))
        mlflow.log_param('final_single_shot_nas_architecture', json_choices)

    logging.info(f'Single-Shot NAS trainning procedure completed. {run_info_msg}')
    return (..., final_architecture_path, architecture_choices)  # TODO: return 'meters' metrics resulting from best evaluation on validset


def handle_nni_nas_trial(training_pipeline_name: str, project_path: Union[str, Path], model: torch.nn.Module, regular_training_fn: TRAINING_PROCEDURE_T, training_kwargs: Dict[str, Any], single_shot_nas_training_fn: TRAINING_PROCEDURE_T = nni_single_shot_neural_architecture_search) -> 'training_procedure_results':
    """ Entry point of any training procedure which may support NNI NAS (single-shot and/or classic NAS) (which itself can be trial-code of an NNI HP search)
    This function will decide whether to trigger regular training procedure, run Single-shot NAS training (training done with appropriate trainer and mutator), run classic NAS trial training (regular training procedure of sampled architecture), generate NNI Classic NAS search space if needed, generate a template for NNI config if missing for this training pipeline (NNI Classic NAS), or apply a fixed architecture reesulting from a previous NNI NAS experiment on given model/training-pipeline.
    NOTE: NNI NAS algorithms can be split into two different categories: Classic NAS and SignleShot NAS algorithms.
    Classic NAS API is much like NNI HP API and needs a YAML NNI Config file and a JSON search space (can be generated from model with mutables)
    while SingleShot NNI NAS is a Python API (not based on `nnictl`) allowing to find best performing architectures in a single training using specific 'trainer'(s) and 'mutator'(s).
    Thus, only NNI SingleShot NAS trainings can be nested within an NNI HP search (i.e. NNI HP trials performing SingleShot NAS instead of regular training).
    If NNI Classic NAS experiment is desired, this function will generate NAS JSON search space and/or NNI config file for Classic NAS if missing, feel free to modify NNI config file generated for this training pipeline according to your needs.
    Args:
        - training_pipeline_name: 
        - model:
        - regular_training_fn:
        - training_kwargs;
        - single_shot_nas_training_fn:
    
    Return output from execution of training procedure (`regular_training_fn` or `single_shot_nas_training_fn`)
    NOTE: NNI SingleShot NAS and regular training procedures should handle `nni.report_final_result` and `nni.report_intermediate_results` calls in case we are performing an NNI HP (or NNI Classic NAS experiment for regular training procedure). You may use `deepcv.meta.nni_tools.is_nni_run_standalone` in training procedure(s) to decide whether to report results to NNI (see `ignite_training.train` training procedure for an example).
    """
    # TODO: take parameters from training_kwargs['hp']?
    nas_algorithm = getattr(getattr(training_kwargs, 'hp', dict()), 'nni_nas_algorithm', None)
    project_path = Path(project_path)

    if model_contains_nni_nas_mutable(model):
        experiment, trial = get_nni_or_mlflow_experiment_and_trial()
        nas_prefix = 'single_shot_nas' if single_shot_nas_algorithm else 'classic_nas'
        exp_str = f'-pipeline_{training_pipeline_name}' if experiment is None else f'-exp_{experiment}'

        # Handle NNI NAS YAML parameters defaults
        conf_base_path = project_path / 'conf' / 'base'
        PARAMS_DEFAULTS = { 'nni_nas_algorithm': None,
                            'classic_nas_config_path': conf_base_path / 'classic_nas_configs' / f'{training_pipeline_name}_nni_config.yml',
                            'common_classic_nas_config_file': conf_base_path / 'classic_nas_configs' / '_common_classic_nas_config.yml',
                            'nni_hp_config_file': conf_base_path / 'nni_hp_configs' / f'{training_pipeline_name}_nni_config.yml',
                            'common_nni_hp_config_file': conf_base_path / 'nni_hp_configs' / '_common_nni_hp_config.yml',
                            'classic_nas_search_space_path': conf_base_path / 'classic_nas_search_spaces' / f'classic_nas_search_space{exp_str}.json',
                            'nni_compression_search_space': conf_base_path / 'nni_compression_spaces' / f'nni_compression_search_space{exp_str}.json',
                            'fixed_architecture_path': project_path / 'data' / '04_training' / f'final-architecture{nas_prefix}{exp_str}{"" if trial is None else f"-trial_{trial}"}.json'}
        nni_nas_params = hyperparams.to_hyperparameters(getattr(training_kwargs, 'hp', dict()), defaults=PARAMS_DEFAULTS)

        if nas_algorithm in NNI_CLASSIC_NAS_ALGORITHMS:
            # If missing generates NNI config and/or NNI search space for Classic NAS experiment
            # TODO: make sure common_nni_config_file is a Path
            pipeline_nni_config_path = nni_nas_params['classic_nas_config_path'] if nni_nas_params['classic_nas_config_path'] is not None else (common_nni_config_file.parent / f'{training_pipeline_name}_nni_config.yml')
            

            if not pipeline_nni_config_path.exists():
                # NNI Classic NAS needs an NNI config file (Single Shot NAS have a pure Python API and wont need an NNI config file, allowing upper/nested NNI HP search) (same for JSON search space)
                logging.info(f'Generating NNI config for NNI Classic NAS for "{training_pipeline_name}" training pipeline.{NL}'
                             f'NOTE: If you want to change NNI config for this pipeline or any other pipelines, please avoid modifying generated config and, instead, modify common NNI config template ("{common_nni_config_file}") or pipeline-specific NAS parameters in `./base/conf/parameters.yml` (overrides common NNI template for this pipeline).')
                gen_nni_config(common_nni_config_file, new_config_path=pipeline_nni_config_path, kedro_pipeline=training_pipeline_name,
                               optimize_mode = 'minimize', hp_tunner = 'TPE', early_stopping='Medianstop')

            # If missing, generate NAS JSON search space from model NNI Mutable(s) for Classic NAS experiment
            if not architecture_search_space_path.exists() and not is_nni_gen_search_space_mode():
                logging.info('Generating JSON search space for NNI Classic NAS from model (Model with NNI NAS mutable `LayerChoice`(s) and/or `InputChoice`(s) converted to a JSON search space for Classic NAS algorithm)')
                # Retreive trial command from NNI Configuration in order to pass it to `nnictl` for JSON NAS search space generation
                nni_config = anyconfig.load(pipeline_nni_config_path)
                training_trial_cmd = nni_config['command']
                code_dir = nni_config['codeDir'] if 'codeDir' in nni_config else None
                # Generate NNI Classic NAS Search space by calling `nnictl ss_gen` automatically
                gen_classic_nas_search_space(architecture_search_space_path, training_trial_cmd, code_dir)
            # Performs NNI Classic NAS sampling over given `model` NAS search space
            # NOTE: If we are performing NNI Classic NAS (not SingleShot NAS), then we are not in a 'upper' NNI HP search (NNI HP search over NNI classic NAS are impossible due to NNI config and `nnictl` usage for both APIs)
            model = nni.nas.pytorch.classic_nas.get_and_apply_next_architecture(model)
            logging.info(f'Sampled a model architecture from NAS search space.{NL}'
                         f'About to perform a trial of NNI Classic NAS for "{training_pipeline_name}" training pipeline...')
        elif nas_algorithm in NNI_SINGLE_SHOT_NAS_ALGORITHMS:
            # Performing NNI Single Shot NAS training procedure (only one trial to generate fixed architecture from NAS)
            logging.info(f'About to perform a Single-Shot NNI NAS for "{training_pipeline_name}" training pipeline..')
            return single_shot_nas_training_fn(**training_kwargs, final_architecture_path=fixed_architecture_path)
        elif nas_algorithm not in (None, r''):
            raise ValueError(f'Error: Bad NNI NAS algorithm name "{nas_algorithm}", valid NNI single-shot NAS algorithms are "{NNI_SINGLE_SHOT_NAS_ALGORITHMS.keys()}" and '
                            f'valid NNI Classic NAS algorithms are "{NNI_CLASSIC_NAS_ALGORITHMS}".')
        elif is_nni_standalone_mode() and fixed_architecture_path.exists():
            # Apply fixed architecture from `fixed_architecture_path` JSON file (best/final architecture choices generated from an NNI NAS experiment)
            # NOTE: If fixed architecture doesnt exists for this model and NNI is in standalone mode, then model will automatically be fixed to the first choice/candidate of any of its Mutable `LayerChoice`/`InputChoice`(s) (just run regular training/trial code)
            logging.info(f'Applying fixed architecture from NNI NAS final results for "{training_pipeline_name}" training pipeline... '
                         f'(fixed_architecture_path="{fixed_architecture_path}")')
            model = nni.nas.pytorch.fixed.apply_fixed_architecture(model, fixed_architecture_path)
    elif nas_algorithm in (None, r'') or is_nni_gen_search_space_mode():
        raise ValueError(f'Error: Cant generate NNI NAS search space nor run an NNI NAS algorithm: `nas_algorithm="{nas_algorithm}"` argument specified while model '
                         f'to be trained doesnt contain any NNI NAS mutable `LayerChoice`(s) nor `InputChoice`(s) (model="{model}")')

    # Run regular training procedure
    logging.info(f'About to perform regular training procedure of "{training_pipeline_name}" training pipeline...')
    return training_procedure(**training_kwargs)


def run_nni_hp_search(pipeline_name, model, backend_conf, hp):
    # TODO: Fill NNI config
    gen_nni_config(...)
    # TODO: Handle NNI NAS Model (Generate HP Space from NNI mutable Layers/inputs in model(s))
    # TODO: merge NNI HP Search space from user with generated NAS Search space
    # TODO: Start NNI process with given pipeline as trial code with nnictl
    cmd = f'nnictl ss-gen-space'


def gen_nni_config(common_nni_config_file: Union[str, Path], new_config_path: Union[str, Path], kedro_pipeline: str, project_ctx: KedroContext, optimize_mode: str = 'minimize', hp_tunner: str = 'TPE', early_stopping: Optional[str] = 'Medianstop', command_opts: Union[str, Sequence[str]] = '') -> bool:
    """ Generates NNI configuration file in order to run an NNI Hyperparameters Search on given pipeline (by default, new `kedro_pipeline`-specific NNI config file will be saved in the same directory as `common_nni_config_file` and will be named after its respective pipeline name, See `handlle_nni_nas_trial`).
    Fills missing NNI configuration fields from defaults/commons NNI YAML config (won't change any existing values from given NNI YAML configuration but appends missing parameters with some defaults).
    It means given NNI YAML config file should only contain parameters which are common to any NNI HP/NAS API usage in DeepCV and this function will populate other parameters according to a specific training pipeline.
    NOTE: `gen_nni_config` wont overwrite any existing NNI configuration named after the same pipeline (i.e., if '{kedro_pipeline}_nni_config.yml' already exists, this function wont do anything).
    .. See [NNI HP API documentation for more details on NNI YAML configuration file](https://nni.readthedocs.io/en/latest/hyperparameter_tune.html)
    """
    common_nni_config_file = Path(common_nni_config_file)

    if not common_nni_config_file.exists():
        msg = f'Error: Couldn\'t find provided NNI config defaults/template/commons at: "{common_nni_config_file}"'
        logging.error(msg)
        raise FileNotFoundError(msg)
    if new_config_path.exists():
        logging.warn(f'Warning: `deepcv.meta.nni_tools.gen_nni_config` called but YAML NNI config file already exists for this pipeline ("{kedro_pipeline}"), '
                     f'"{new_config_path.name}" YAML config wont be modified, you may want to delete it before if you need to update it.{NL}'
                     f'Also note that you can customize "{common_nni_config_file}" config if you need to change NNI common behavior for any NNI HP/NAS API usage in DeepCV (Hyperparameter searches and Neural Architecture Searches based on NNI); All NNI configuration are generated from this template/common/default YAML config. See also "deepcv.meta.nni_tools.gen_nni_config" function for more details about NNI config handling in DeepCV.')
        return False

    experiment, trial = get_nni_or_mlflow_experiment_and_trial()
    nni_config = anyconfig.load(common_nni_config_file, ac_parser='yaml')

    nni_config['authorName'] = getattr(nni_config, 'authorName', __author__)
    nni_config['experimentName'] = getattr(nni_config, 'experimentName', (experiment if experiment not in (None, '') else f'{project_ctx.project_name}_{kedro_pipeline}'.lower()))
    nni_config['searchSpacePath'] = getattr(nni_config, 'searchSpacePath', common_nni_config_file.parent / f'hp_search_spaces/{kedro_pipeline}_search_space.json')
    nni_config['trialConcurrency'] = getattr(nni_config, 'trialConcurrency', 1)
    nni_config['maxTrialNum'] = getattr(nni_config, , -1)
    nni_config['trainingServicePlatform'] = getattr(nni_config, 'trainingServicePlatform', 'local')

    trial_conf = nni_config['trial'] if 'trial' in nni_config else dict()
    trial_conf['command'] = getattr(trial_conf, 'command', f'kedro run --pipeline={kedro_pipeline} {command_opts if isinstance(command_opts, str) else " ".join(command_opts)}')
    trial_conf['codeDir'] = getattr(trial_conf, 'codeDir', common_nni_config_file / r'../../src/deepcv')
    trial_conf['gpuNum'] = getattr(trial_conf, 'gpuNum', 0)
    nni_config['trial'] = trial_conf

    tuner_conf = nni_config['tuner'] if 'tuner' in nni_config else dict()
    tuner_conf['builtinTunerName'] = getattr(tuner_conf, 'builtinTunerName', hp_tunner)
    tuner_conf['classArgs'] = getattr(tuner_conf, 'classArgs', {'optimize_mode': optimize_mode})
    nni_config['tuner'] = tuner_conf

    if early_stopping is not None:
        assesor_conf = nni_config['assessor'] if 'assessor' in nni_config else dict()
        assesor_conf['builtinAssessorName'] = getattr(assesor_conf, 'builtinAssessorName', early_stopping)
        assesor_conf['classArgs'] = getattr(assesor_conf, 'classArgs', {'optimize_mode': optimize_mode, 'start_step': 8})
        nni_config['assessor'] = assesor_conf

    # Save final NNI configuration as a new YAML file named after its respective training pipeline
    anyconfig.dump(nni_config, new_config_path, ac_parser='yaml')
    return True


def sample_nni_hp_space(model_hps: HYPERPARAMS_T, training_hps: HYPERPARAMS_T) -> Tuple[HYPERPARAMS_T, HYPERPARAMS_T]:
    """ Sample hyperparameters from NNI search space and merge those with given model definition and training procedure hyperparameters (which are probably from YAML config) """
    params_from_nni = nni.get_next_parameter()

    # Fill values sampled from NNI seach space in their respective hyperparameters set (model_hps or training_hps)
    for name, value in params_from_nni.items():
        is_model = name.startswith('model:')
        if not is_model and not name.startswith('training:'):
            raise ValueError('Error: NNI hyperparameter names should either start with `training:` or `model:` to specify whether parameter belongs to training procedure or model definition.')

        # Recursive call to dict.__getitem__, which allows to access nested parameters by using a `.` between namespaces
        *hierachy, parameter_name = name.split('.')[1:]
        functools.reduce(dict.__getitem__, [model_hps if is_model else training_hps, *hierachy])[parameter_name] = value

    return model_hps, training_hps


def hp_search(hp_space: Dict[str, Any], define_model_fn: Callable[['hp'], torch.nn.Module], training_procedure: Callable, dataloaders: Tuple[DataLoader], pred_across_scales: bool = False, subset_sizes: Sequence[float] = [0.005, 0.015, 0.03, 0.05, 0.07, 0.09]):
    """ NNI Hyperparameter search trial procedure, with optional generalization prediction across scales """
    model_hps, training_hps = sample_nni_hp_space(model_hps=..., training_hps=...)  # TODO: finishn implementation of hp_search
    logging.info(f'> Hyperparameter search experiment "{nni.get_experiment_id()}" -- trial NO#{nni.get_trial_id()}...{NL}'
                 f'sampled training_hps="{training_hps}"{NL}'
                 f'sampled model_hps="{model_hps}"')
    trainset, *validset_and_testset = dataloaders[0], dataloaders[1:]

    # Create model from `define_model_fn`
    model = define_model_fn(hp=model_hps)

    if pred_across_scales:
        logging.info('Using "prediction of model\'s generalization across scales" technique by performing multiple trainings on small trainset subsets to be able to predict validation error if we had trained model on full trainset.')
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        dl_kwargs = dict(batch_size=training_hps['batch_size'], num_workers=num_workers, pin_memory=True)
        subsets = [datasets.get_random_subset_dataloader(trainset, ss, **dl_kwargs) for ss in subset_sizes]
        generalization_predictor = hyperparams.GeneralizationAcrossScalesPredictor(
            len(subsets), fit_using_hps=False, fit_using_loss_curves=False, fit_using_dataset_stats=False)

    # TODO: add ignite training handler to report metrics to nni as intermediate results: nni.report_intermediate_result()

    if pred_across_scales:
        best_valid_losses_across_scales = [training_procedure(model, training_hps, trainset=s, validset=validset_and_testset[0]) for s in subsets]
        models = [model, ] * len(best_valid_losses_across_scales)  # We always train the same model across different trainset subset sizes
        generalization_predictor.fit_generalization(subsets, models, best_valid_losses_across_scales)
        predicted_score = generalization_predictor(model, trainset)  # TODO: get trainset as parameter
        nni.report_final_result(predicted_score)
    else:
        rslt = training_procedure(model, training_hps, trainset=trainset, validset=validset_and_testset[0])
        nni.report_final_result(rslt)

    logging.info(f'######## NNI Hyperparameter search trial NO#{nni.get_trial_id()} done! ########')


def get_hp_position_in_search_space(hp, hp_search_space):
    # TODO: return hp set position in given search space (aggregates position of each searchable parameters in their respective search range, nested choices/sampling multiplies its parent relative position)
    raise NotImplementedError
    return torch.zeros((1, 10))


def generate_hp_space_template(hyperparams: HYPERPARAMS_T, save_filename: str = r'nni_hp_space_template.json', exclude_params: Sequence[str] = None, include_params: Sequence[str] = None) -> Tuple[Dict[str, Any], Path]:
    """ Generates an hyperparameter space template/draft from given hyperparameter set, making it easier to define your JSON (NNI spec.) hp space definition (generates a start point for your JSON hp space)
    # TODO: active learning of a lightweight model which infers (options, low, high, q, mu and/or sigma) from (parameter name, type, hierarchy/context)
    # TODO: Samples from hp space should be merged to `hp` dict from YAML and vis versa, a given `hp` set should have a position in an hp search space
    """
    options, low, high, q, mu, sigma = list(), 0., 1., 1, 0., 1.  # q is the discrete step size
    nni_search_space_specs = [{'_type': 'choice', '_value': options},
                              # `randint` is equivalent to `quniform` with `q = 1` but may be interpreted as unordered by some NNI hp search tuners while `quniform` is always ordered
                              {'_type': 'randint', '_value': [low, high]},
                              {'_type': 'uniform', '_value': [low, high]},
                              {'_type': 'quniform', '_value': [low, high, q]},
                              {'_type': 'loguniform', '_value': [low, high]},
                              {'_type': 'qloguniform', '_value': [low, high, q]},
                              {'_type': 'normal', '_value': [mu, sigma]},
                              {'_type': 'qnormal', '_value': [mu, sigma, q]},
                              {'_type': 'lognormal', '_value': [mu, sigma]},
                              {'_type': 'qlognormal', '_value': [mu, sigma, q]}]
    search_space_types = {spec['_type'] for spec in nni_search_space_specs}

    exclude_params = set() if exclude_params is None else set(exclude_params)
    include_params = set() if include_params is None else set(include_params)
    hp_search_space = {n: dict(_type=v, _value=[v]) for n, v in hyperparams.items() if n in include_params and n not in exclude_params}
    hp_search_space_json = Path.cwd() / save_filename

    # Save hyperparameters search space template as JSON file
    with open(hp_search_space_json, 'w') as json_file:
        json.dump(hp_search_space, json_file)

    return hp_search_space, hp_search_space_json
    # "dropout_keepprob": {"_type": "uniform", "_value": [0.1, 0.5]},
    # "conv_size": {"_type": "choice", "_value": [2, 3, 5, 7]},
    # "hidden_size": {"_type": "choice", "_value": [124, 512, 1024]},
    # "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128, 256]},
    # "lr": {"_type": "uniform", "_value": [0.0001, 0.1]}
    # }

#_____________________________________________ NNI INTEGRATION UNIT TESTS _____________________________________________#


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
