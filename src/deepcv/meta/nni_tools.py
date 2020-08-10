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
from typing import Sequence, Iterable, Callable, Dict, Tuple, Any, Union, Optional, List, Type

import mlflow
import anyconfig

import torch
import torch.nn as nn
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

__all__ = ['is_nni_single_shot_nas_algorithm', 'is_nni_classic_nas_algorithm', 'is_nni_gen_search_space_mode', 'is_nni_run_standalone', 'get_nni_or_mlflow_experiment_and_trial', 'model_contains_nni_nas_mutable',
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


def is_nni_single_shot_nas_algorithm(algorithm: str) -> bool:
    if algorithm in NNI_SINGLE_SHOT_NAS_ALGORITHMS:
        return True
    if algorithm in NNI_CLASSIC_NAS_ALGORITHMS:
        return False
    raise ValueError(f'Error: Bad NNI NAS algorithm name "{algorithm}", valid NNI SingleShot NAS algorithms are "{NNI_SINGLE_SHOT_NAS_ALGORITHMS.keys()}" and '
                     f'valid NNI Classic NAS algorithms are "{NNI_CLASSIC_NAS_ALGORITHMS}".')


def is_nni_classic_nas_algorithm(algorithm: str) -> bool:
    return not is_nni_single_shot_nas_algorithm(algorithm)


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


def nni_single_shot_neural_architecture_search(hp: HYPERPARAMS_T, model: nn.Module, loss: Union[nn.modules.loss._Loss, Callable], datasets: Tuple[Dataset], opt: Type[torch.optim.Optimizer], backend_conf: 'ignite_training.BackendConfig' = None,
                                               metrics: Dict[str, METRIC_FN_T] = {}, callbacks_handler: deepcv.utils.EventsHandler = None, final_architecture_path: Union[str, Path] = None) -> Tuple[METRICS_DICT_T, Optional[Path], str]:
    """ Train model with provided NAS trainer in order to find out the best NN architecture by training a superset NN instead of performing multiple trainings/trials for each/many possible architectures.
    Args:
        - hp:
        - model:
        - loss:
        - datasets:
        - opt:
        - backend_conf:
        - metrics:
        - callbacks_handler:
        - final_architecture_path:
    Returns a pathlib.Path to a JSON file storing the best NN model architecture found by NNI Single-Shot NAS (JSON file storing mutable layer(s)/input(s) choices made in model search space in order to define best fixed architeture found; This file is also logged to mlflow if there is an active run)
    NOTE: Support for SPOS SingleShot NAS is partial for now, see [NNI NAS SPOS documentation](https://nni.readthedocs.io/en/latest/NAS/SPOS.html) for more details on Evolution Tuner looking for best architecture and its usage. (be aware of issues around batch normalization which should be retrained before evaluation when using SPOS Evolution)
    # TODO: convert ignite metrics for NNI NAS trainer usage if needed (to Callable[['outout', 'target'], Dict[str, float]])
    # TODO: reuse code from ignite training for output path and log final architecture as mlflow artifact
    # TODO: Allow resuming an NNI single shot NAS experiment throught 'hp['resume_from']' parameter (if possible easyly using NNI API?)
    # TODO: Add support for two-way callbacks using deepcv.utils.EventsHandler in a similar way than ignite_training.train (once ignite_training fully support it)
    """
    from .ignite_training import BackendConfig
    if backend_conf is None:
        backend_conf = BackendConfig()
    experiment_name, run_id = get_nni_or_mlflow_experiment_and_trial()
    run_info_msg = f'(Experiment: "{experiment_name}", run_id: "{run_id}")'
    logging.info(f'Starting Single-Shot Neural Architecture Search (NNI NAS API) training over NN architecture search space {run_info_msg}.')

    TRAINING_HP_DEFAULTS = {'optimizer_opts': ..., 'epochs': ..., 'batch_size': None, 'nni_single_shot_nas_algorithm': ..., 'output_path': Path.cwd() / 'data/04_training/', 'log_output_dir_to_mlflow': True,
                            'log_progress_every_iters': 100, 'seed': None, 'resume_from': '', 'deterministic_cudnn': False, 'nas_mutator': None, 'nas_mutator_kwarg': dict(), 'nas_trainer_kwargs': dict()}
    hp, _ = hyperparams.to_hyperparameters(hp, TRAINING_HP_DEFAULTS, raise_if_missing=True)
    deepcv.utils.setup_cudnn(deterministic=hp['deterministic_cudnn'], seed=backend_conf.rank + hp['seed'])  # In distributed setup, we need to have different seed for each workers
    model = model.to(backend_conf.device, non_blocking=True)
    loss = loss.to(backend_conf.device) if isinstance(loss, torch.nn.Module) else loss
    num_workers = max(1, (backend_conf.ncpu - 1) // (backend_conf.ngpus_current_node if backend_conf.ngpus_current_node > 0 and backend_conf.distributed else 1))
    optimizer = opt(model.parameters(), **hp['optimizer_opts'])
    trainset, *validset_testset = datasets

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
    if final_architecture_path is not None:
        logging.info(f'Saving final/best NN architeture obtained from NNI Single-Shot NAS (and may log it to MLFLow artifacts). {run_info_msg}')
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


def handle_nni_nas_trial(training_pipeline_name: str, model: torch.nn.Module, regular_training_fn: TRAINING_PROCEDURE_T, training_kwargs: Dict[str, Any], single_shot_nas_training_fn: TRAINING_PROCEDURE_T = nni_single_shot_neural_architecture_search) -> 'training_procedure_results':
    """ Entry point of any training procedure which may support NNI NAS (single-shot and/or classic NAS) (which itself can be trial-code of an NNI HP search)
    This function will decide whether to trigger regular training procedure, run Single-shot NAS training (training done with appropriate trainer and mutator), run classic NAS trial training (regular training procedure of sampled architecture), generate NNI Classic NAS search space if needed, generate a template for NNI config if missing for this training pipeline (NNI Classic NAS), or apply a fixed architecture reesulting from a previous NNI NAS experiment on given model/training-pipeline.
    NOTE: NNI NAS algorithms can be split into two different categories: Classic NAS and SignleShot NAS algorithms.
    Classic NAS API is much like NNI HP API and needs a YAML NNI Config file and a JSON search space (can be generated from model with mutables)
    while SingleShot NNI NAS is a Python API (not based on `nnictl`) allowing to find best performing architectures in a single training using specific 'trainer'(s) and 'mutator'(s).
    Thus, only NNI SingleShot NAS trainings can be nested within an NNI HP search (i.e. NNI HP trials performing SingleShot NAS instead of regular training).
    If NNI Classic NAS experiment is desired, this function will generate NAS JSON search space and/or NNI config file for Classic NAS if missing, feel free to modify NNI config file generated for this training pipeline according to your needs.
    NOTE: NNI SingleShot NAS and regular training procedures should handle `nni.report_final_result` and `nni.report_intermediate_results` calls in case we are performing an NNI HP (or NNI Classic NAS experiment for regular training procedure). You may use `deepcv.meta.nni_tools.is_nni_run_standalone` in training procedure(s) to decide whether to report results to NNI (see `ignite_training.train` training procedure for an example).
    """

    if model_contains_nni_nas_mutable(model):
        experiment, trial = get_nni_or_mlflow_experiment_and_trial()
        nas_prefix = 'single_shot_nas' if single_shot_nas_algorithm else 'classic_nas'
        exp_str = f'-pipeline_{training_pipeline_name}' if experiment is None else f'-exp_{experiment}'

        # Determine `fixed_architecture_path` file path
        if specified_fixed_architecture is not None:
            fixed_architecture_path = Path(specified_fixed_architecture)
        else:
            # Default NAS fixed/final architecture better should be named after trial/run ID if we are performing an NNI SingleShot NAS within an NNI HP search (Nested usage of NNI HP and NNI SingleShot NAS APIs: MLFlow run ID is named after 'upper' NNI HP search trial ID)
            fixed_architecture_path = Path(hp['output_path']) / f'{nas_prefix}{exp_str}{"" if trial is None else f"-trial_{trial}"}.json'

        if is_nni_classic_nas_algorithm(hp['nni_nas_algorithm']):
            # If missing generates NNI config and/or NNI search space for Classic NAS experiment
            # TODO: make sure common_nni_config_file is a Path
            pipeline_nni_config_path = common_nni_config_file.parent / f'{kedro_pipeline}_nni_config.yml'
            architecture_search_space_path = Path.cwd() / ... / f'{nas_prefix}_search_space{exp_str}.json'

            if not pipeline_nni_config_path.exists():
                # NNI Classic NAS needs an NNI config file (Single Shot NAS have a pure Python API and wont need an NNI config file, allowing upper/nested NNI HP search) (same for JSON search space)
                logging.info(f'Generating NNI config for NNI Classic NAS for "{training_pipeline_name}" training pipeline.{NL}'
                             f'NOTE: If you want to change NNI config for this pipeline or any other pipelines, please avoid modifying generated config and, instead, modify common NNI config template ("{common_nni_config_file}") or pipeline-specific NAS parameters in `./base/conf/parameters.yml` (overrides common NNI template for this pipeline).')
                gen_nni_config(common_nni_config_file, pipeline_nni_config_path, training_pipeline_name, ...)

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
        elif is_nni_single_shot_nas_algorithm(hp['nni_nas_algorithm']):
            # Performing NNI Single Shot NAS training procedure (only one trial to generate fixed architecture from NAS)
            logging.info(f'About to perform a Single-Shot NNI NAS for "{training_pipeline_name}" training pipeline..')
            return single_shot_nas_training_fn(**training_kwargs, final_architecture_path=fixed_architecture_path)
        elif is_nni_standalone_mode() and fixed_architecture_path.exists():
            # Apply fixed architecture from `fixed_architecture_path` JSON file (best/final architecture choices generated from an NNI NAS experiment)
            # NOTE: If fixed architecture doesnt exists for this model and NNI is in standalone mode, then model will automatically be fixed to the first choice/candidate of any of its Mutable `LayerChoice`/`InputChoice`(s) (just run regular training/trial code)
            logging.info(
                f'Applying fixed architecture from NNI NAS final results for "{training_pipeline_name}" training pipeline... (fixed_architecture_path="{fixed_architecture_path}")')
            model = nni.nas.pytorch.fixed.apply_fixed_architecture(model, fixed_architecture_path)
    elif asked_for_nas or is_nni_gen_search_space_mode():
        raise ValueError(
            f'Error: Cant generate NNI NAS search space nor run an NNI NAS Algorithm ("{...}" parameters specified) while model to be trained doesnt contain any NNI NAS mutable `LayerChoice`(s) nor `InputChoice`(s) (model="{model}")')

    # Run regular training procedure
    return training_procedure(**training_kwargs)


def run_nni_hp_search(pipeline_name, model, backend_conf, hp):
    # TODO: Fill NNI config
    gen_nni_config(...)
    # TODO: Handle NNI NAS Model (Generate HP Space from NNI mutable Layers/inputs in model(s))
    # TODO: merge NNI HP Search space from user with generated NAS Search space
    # TODO: Start NNI process with given pipeline as trial code with nnictl
    cmd = f'nnictl ss-gen-space'


def gen_nni_config(common_nni_config_file: Union[str, Path], new_config_path: Union[str, Path], kedro_pipeline: str, optimize_mode: str = 'minimize', hp_tunner: str = 'TPE', early_stopping: Optional[str] = 'Medianstop'):
    """ Generates NNI configuration file in order to run an NNI Hyperparameters Search on given pipeline (new/full NNI config file will be save in the same directory as `common_nni_config_file` and will be named after its respective pipeline name).
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
        return

    nni_config = anyconfig.load(common_nni_config_file, ac_parser='yaml')

    def _set_parameter_if_not_defined(nni_config, parameter_name: str, default_value: Any):
        nni_config[parameter_name] = getattr(nni_config, parameter_name, default_value)

    _set_parameter_if_not_defined(nni_config, 'authorName', __author__)
    _set_parameter_if_not_defined(nni_config, 'experimentName', nni.get_experiment_id())
    _set_parameter_if_not_defined(nni_config, 'searchSpacePath', common_nni_config_file / f'hp_search_spaces/{kedro_pipeline}_search_space.json')
    _set_parameter_if_not_defined(nni_config, 'trialConcurrency', 1)
    _set_parameter_if_not_defined(nni_config, 'maxTrialNum', -1)
    _set_parameter_if_not_defined(nni_config, 'trainingServicePlatform', 'local')

    trial_conf = nni_config['trial'] if 'trial' in nni_config else dict()
    _set_parameter_if_not_defined(trial_conf, 'command', f'kedro run --pipeline={kedro_pipeline}')
    _set_parameter_if_not_defined(trial_conf, 'codeDir', common_nni_config_file / r'../../src/deepcv')
    _set_parameter_if_not_defined(trial_conf, 'gpuNum', 0)
    nni_config['trial'] = trial_conf

    tuner_conf = nni_config['tuner'] if 'tuner' in nni_config else dict()
    _set_parameter_if_not_defined(tuner_conf, 'builtinTunerName', hp_tunner)
    _set_parameter_if_not_defined(tuner_conf, 'classArgs', {'optimize_mode': optimize_mode})
    nni_config['tuner'] = tuner_conf

    if early_stopping is not None:
        assesor_conf = nni_config['assessor'] if 'assessor' in nni_config else dict()
        _set_parameter_if_not_defined(assesor_conf, 'builtinAssessorName', early_stopping)
        _set_parameter_if_not_defined(assesor_conf, 'classArgs', {'optimize_mode': optimize_mode, 'start_step': 8})
        nni_config['assessor'] = assesor_conf

    # Save final NNI configuration as a new YAML file named after its respective training pipeline
    anyconfig.dump(nni_config, new_config_path, ac_parser='yaml')


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


def hp_search(hp_space: Dict[str, Any], define_model_fn: Callable[['hp'], nn.Module], training_procedure: Callable, dataloaders: Tuple[DataLoader], pred_across_scales: bool = False, subset_sizes: Sequence[float] = [0.005, 0.015, 0.03, 0.05, 0.07, 0.09]):
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
