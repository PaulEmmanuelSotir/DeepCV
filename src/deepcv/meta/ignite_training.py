#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Training loop meta module - training_loop.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
    - TODO: Implement an `eval_on_testset` function in order to perform a final evaluation of best perfoming model on testset after training procdeure
"""
import shutil
import logging
import traceback
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Iterable, Tuple, Union, Type, Callable, Sequence, Mapping

import torch
import torch.nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

import ignite
import ignite.utils
import ignite.metrics
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.contrib.engines import common
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler, GradsHistHandler
import ignite.contrib.handlers

import mlflow
import kedro
import nni

import deepcv.utils
from .nn import is_data_parallelization_usefull_heuristic, data_parallelize
from .nni_tools import is_nni_run_standalone, get_nni_or_mlflow_experiment_and_trial
from .hyperparams import to_hyperparameters
from .types_aliases import *

from .data.datasets import dataloader_prefetch_batches

__all__ = ['MAIN_TRAINING_LOSS_NAME', 'HyperparamsOutpoutHandler', 'BackendConfig', 'train']
__author__ = 'Paul-Emmanuel Sotir'

# Default loss name (prefix) used for training loss obtained from ponderated mean of each provided loss terms (see `losses` and `loss_weights` arguments of `deepcv.meta.ignite_training.train` training procedure) 
MAIN_TRAINING_LOSS_NAME = 'main_loss'


class HyperparamsOutpoutHandler(BaseOutputHandler):
    """ Custom ignite output handler for tensorboard hyperparameter set logging  
    .. See [PyTorch tensorboard support for more details on hyperparameter set logging](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams)  
    (Preferaly use it for HP search and/or on Events.COMPLETE ignite events)
    """
    def __init__(self, hp: Union[Dict[str, Any], deepcv.meta.hyperparams.Hyperparameters], tag = 'hpparam', metric_names: Sequence[str] = None, output_transform: Callable = None):
        super(HyperparamsOutpoutHandler, self).__init__(tag, metric_names, output_transform, None)
        self.hp = hp.get_dict_view() if isinstance(hp, deepcv.meta.hyperparams.Hyperparameters) else hp
    
    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: str):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError(f'Error: Handler `{type(self).__name__}` works only with `TensorboardLogger` logger')

        metrics = self._setup_output_metrics(engine)

        prefixed_metrics = dict()
        for key, value in metrics.items():
            if isinstance(value, numbers.Number) or isinstance(value, torch.Tensor) and value.ndimension() == 0:
                prefixed_metrics['{self.tag}/{key}'] = value
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for i, v in enumerate(value):
                    prefixed_metrics[f'{self.tag}/{key}/{i}'] = v.item()
            else:
                warnings.warn(f'Warning: TensorboardLogger {type(self).__name__} can not log metrics value type {type(value)}')
        logger.writer.add_hparams(self.hp, metric_dict={f'hpparam/{m}': v for m, v in metrics}})


class BackendConfig:
    """ Training backend configuration for device and distributed training setup
    # TODO: Refactor and clean this after having read pytorh docs and examples: https://pytorch.org/docs/stable/distributed.html#distributed-launch
    """

    def __init__(self, device_or_id: Union[None, str, int, torch.device] = None, dist_backend: dist.Backend = None, dist_url: str = None):
        if device_or_id is None:
            self.device = deepcv.utils.get_device()
        elif isinstance(device_or_id, (str, torch.device)):
            self.device = torch.device(device_or_id) if isinstance(device_or_id, str) else device_or_id
        else:
            self.device = deepcv.utils.get_device(devid=device_or_id)  # `device_or_id` is assumed to be an `int` (device ID)
        self.is_cpu = self.device.type == 'cpu'
        self.is_cuda = self.device.type == 'cuda'  # Possible devices as of PyTorch 1.6.0: cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu, xla
        self.ncpu = multiprocessing.cpu_count()
        self.dist_backend = dist_backend
        self.distributed = dist_backend is not None
        self.dist_url = dist_url
        self.local_rank = getattr(self.device, 'index', None)
        self.ngpus_current_node = torch.cuda.device_count()
        self.rank, self.nnodes = 0, 1  # `rank` is dist rank here (not GPU device ID/index `local_rank`)

        if self.distributed:
            self.rank = dist.get_rank()
            self.gpus_world_size = dist.get_world_size()
            self.nnodes = dist.get_world_size() // self.ngpus_current_node  # TODO: fix issue: self.nnodes correct only if each nodes have the same GPU count

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if self.is_cpu:
            return f'single-node-cpu-{self.ncpu}'
        if self.distributed:
            return f'distributed-{self.nnodes}avg_nodes-{self.gpus_world_size}gpus_world_size-{self.ngpus_current_node}current-node-gpus(rank={self.rank})'
        return f'single-node-{self.ngpus_current_node}-available-gpus'

    @property
    def distributed(self):
        return self.dist_backend is not None and self.dist_backend != '' and self.dist_url is not None and self.dist_url != ''


# TODO: Fire various events with two-way callbacks using `deepcv.utils.EventsHandler` at every steps of training procedure, including underlying ignite events
TRAINING_EVENTS = {'TRAINING_INIT', 'AFTER_TRAINING_INIT', 'ON_EPOCH_STARTED', 'ON_EPOCH_COMPLETED',
                   'ON_ITERATION_STARTED', 'ON_ITERATION_COMPLETED', 'ON_EVAL_STARTED', 'ON_EVAL_COMPLETED', 'ON_TRAINING_COMPLETED'}

def add_training_output_dir(output_path: Union[str, Path], backend_conf: BackendConfig = None, prefix: str = '', named_after_datetime_now: bool =True) -> Path:
    """ Determine output directory name and create it if it deosn exists yet (Aimed at training experiment output directory) """
    if backend_conf is None or not backend_conf.distributed or backend_conf.rank == 0:
        now = '-' + datetime.now().strftime(r'%Y_%m_%d-%HH_%Mmin') if named_after_datetime_now else ''
        backend_conf = '' if backend_conf is None else f'-{str(backend_conf)}'
        experiment, trial = get_nni_or_mlflow_experiment_and_trial()
        if experiment:
            output_path = Path() / f'{prefix}exp_output_{experiment}_run_{trial}{now}{backend_conf}'
        else:
            output_path = Path(hp['output_path']) / f'{prefix}exp_output{now}{backend_conf}'
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


def _setup_ignite_losses(losses: LOSS_FN_TERMS_T, loss_weights: Sequence[FLOAT_OR_FLOAT_TENSOR_T] = None, device: Union[str, torch.device] = None, default_name_prefix: str = 'loss_term_', main_training_loss_name: str = MAIN_TRAINING_LOSS_NAME) -> Dict[str, ignite.metrics.Loss]:
    """ Check and cast given `losses` loss term(s) to output new `ignite.metrics.Loss` object(s) mapped to their name and eventually ponderated by their repective weight from `loss_weights`. """
    def _ignite_loss_get_batchsize(target_tensor: TENSOR_OR_SEQ_OF_TENSORS_T) -> int:
        """ Function used by ignite to gt batch size from tensor(s). This non-default callable is needed in case of sequence of tensor(s) usage instead of a single tensor (`TENSOR_OR_SEQ_OF_TENSORS_T`). """
        if deepcv.utils.is_torch_obj(target_tensor):
            return len(target_tensor) # Assume target_tensor is a single tensor
        elif len(target_tensor) > 0:
            return len(target_tensor[0]) # Assume target_tensor is a sequence of mutliple tensors of the same batch size
        return 0

    # Handle different accepted `losses` types and check validity/coherence of given arguments
    if loss_weights is not None and (isinstance(loss_weights, Mapping) != isinstance(losses, Mapping) or getattr(losses, '__len__', lambda: 1)() != getattr(loss_weights, '__len__', lambda: 1)()):
        raise TypeError(f'Error: `loss_weights` and `losses` should either both be mapping/dicts or both sequences of the same size (or single loss term and weight).{deepcv.utils.NL}'
                        f'Got `loss_weights="{loss_weights}"` and `losses="{losses}"`')
    if not isinstance(losses, (Sequence, Mapping)):
        losses = {main_training_loss_name: (losses, 1.),}
    elif isinstance(losses, Sequence):
        losses = {(f'{default_name_prefix}{i}' if len(losses) > 1 else main_training_loss_name): (loss, w) for i, (loss, w) in enumerate(zip(losses, loss_weights if loss_weights is not None and len(losses) > 1 else [1.,] * len(losses)))}
    else: # Otherwise, assume `losses` and `loss_weights` are mappings
        losses = {n: (v, loss_weights[n] if loss_weights is not None and len(losses) > 1 else 1.) for n, v in losses.items()}
    
    # Instanciate new `ignite.metrics.Loss` objects and make sure to multiply each loss terms by ther respective weight and to divide it by the sum of all weight(s)
    loss_weights_sum = torch.sum(list(zip(*losses.values()))[1]).to(device) # Sum of loss term(s) weights is needed to divide ponderated sum of loss terms when processing final training loss (which makes it a ponderated mean)
    ignite_losses = dict()
    for n, (loss, w) in losses.items():
        if isinstance(loss, torch.nn.Module):
            loss = loss.to(device)
        w = w.to(device) if isinstance(w, torch.Tensor) else torch.FloatTensor(w, device=device)
        if not (w / loss_weights_sum).eq(1.):
            loss = lambda *args, **kwargs: w * loss(*args, **kwargs)
        if not isinstance(loss, ignite.metrics.Loss):
            loss = ignite.metrics.Loss(loss, device=device, batch_size=_ignite_loss_get_batchsize)
        ignite_losses[n] = loss

    # If multiple loss terms are provided, we add `main_training_loss_name` loss which is the ponderated sum of each provided loss terms 
    if main_training_loss_name not in ignite_losses and len(ignite_losses) > 1:
        def _training_loss(*args, **kwargs): return torch.sum(ignite_losses.values(), dim=1) / loss_weights_sum
        ignite_losses[main_training_loss_name] = ignite.metrics.Loss(_training_loss, device=device, batch_size=_ignite_loss_get_batchsize)
    return ignite_losses

def train(hp: HYPERPARAMS_T, model: torch.nn.Module, losses: LOSS_FN_TERMS_T, datasets: Tuple[Dataset], opt: Type[torch.optim.Optimizer], backend_conf: BackendConfig = BackendConfig(),
          loss_weights: Sequence[FLOAT_OR_FLOAT_TENSOR_T] = None, metrics: Dict[str, METRIC_FN_T] = {}, callbacks_handler: deepcv.utils.EventsHandler = None, nni_compression_pruner: nni.compression.torch.Compressor = None) -> Tuple[METRICS_DICT_T, ignite.engine.State]:
    """ Pytorch model training procedure defined using ignite  

    Args:
        - hp: Hyperparameter dict, see ```deepcv.meta.ignite_training._check_params`` to see required and default training (hyper)parameters  
        - model: Pytorch ``torch.nn.Module`` to train  
        - losses: Loss(es) module(s) and/or callables to be used as training criterion(s) (may be a single loss function/module, a sequence of loss functions or a mapping of loss function(s) assiciated to their respective loss name(s)).  
            .. See `loss_weights` argument for more details on multiple loss terms usage and their weighting.
        - datasets: Tuple of pytorch Dataset giving access to trainset, validset and an eventual testset  
        - opt: Optimizer type to be used for gradient descent  
        - backend_conf: Backend information defining distributed configuration (available GPUs, whether if CPU or GPU are used, distributed node count, ...), see ``deepcv.meta.ignite_training.BackendConfig`` class for more details.  
        - loss_weights: Optional weights/factors sequence or mapping to be applied to each loss terms (`loss_weights` defaults to `[1.,] * len(losses)` when `None` or when `len(losses) <= 1`). This argument should contain as many scalars as there are loss terms/functions in `losses` argument. Sum of all weights should be different from zero due to the mean operator (loss terms ponderated sum is divided by the sum/L1-norm of their weights).  
            NOTE: Each scalar in `loss_weights` weights/multiplies its respective loss term so that the total loss on which model is trained is the ponderated mean of each loss terms (e.g. when `loss_weights` only contains `1.` values for each loss term, then the final loss is the mean of each of those).  
            NOTE: You may provide a mapping of weights instead of a Sequence in case you need to apply weights/factors to their respective term identified by their names (`losses` should also be a mapping in this case).  
        - metrics: Additional metrics dictionnary (loss is already included in metrics to be evaluated by default)  
        - callbacks_handler: Callbacks Events handler. If not `None`, events listed in `deepcv.meta.ignite_training.TRAINING_EVENTS` will be fired at various steps of training process, allowing to extend `deepcv.meta.ignite_training.train` functionnalities ("two-way" callbacks ).  
        - nni_compression_pruner: Optional argument for NNI compression support. If needed, provide NNI Compressor object used to prune or quantize model during training so that compressor will be notified at each epochs and steps. See NNI Compression docs. for more details: https://nni.readthedocs.io/en/latest/Compressor/QuickStart.html  
    Returns a [`ignite.engine.state`](https://pytorch.org/ignite/engine.html#ignite.engine.State) object which describe ignite training engine's state (iteration, epoch, dataloader, max_epochs, metrics, ...).  

    NOTE: `callbacks_handler` argument makes possible to register "two-way callbacks" for events listed in `deepcv.meta.ignite_training.TRAINING_EVENTS`; Systematic "two-way callbacks" usage in training loop is a pattern well described in [FastAI blog post about DeepLearning API implementation choices for better flexibility and code stability using multiple API layers/levels and two-way callbacks](https://www.fast.ai/2020/02/13/fastai-A-Layered-API-for-Deep-Learning/#callback). "Two-way callback" usage at every steps of training procedure allows easier modification of training loop behavior without having to modify its code, which allows to implement new features while preserving code stability.  
    # TODO: add support for cross-validations  
    """
    TRAINING_HP_DEFAULTS = {'optimizer_opts': ..., 'epochs': ..., 'scheduler': None, 'output_path': Path.cwd() / 'data/04_training/', 'log_output_dir_to_mlflow': True,
                            'validate_every_epochs': 1, 'save_every_iters': 1000, 'log_grads_every_iters': -1, 'log_progress_every_iters': 100, 'seed': None,
                            'prefetch_batches': True, 'resume_from': '', 'crash_iteration': -1, 'deterministic_cudnn': False, 'use_sync_batch_norm': False}
    logging.info(f'Starting ignite training procedure to train "{model}" model...')
    assert len(datasets) == 3 or len(datasets) == 2, 'Error: datasets tuple must either contain: `trainset and validset` or `trainset, validset and testset`'
    hp, _ = to_hyperparameters(hp, TRAINING_HP_DEFAULTS, raise_if_missing=True)
    device = backend_conf.device
    deepcv.utils.setup_cudnn(deterministic=hp['deterministic_cudnn'], seed=backend_conf.rank + hp['seed'])  # In distributed setup, we need to have different seed for each workers

    # Create dataloaders from given datasets
    num_workers = max(1, (backend_conf.ncpu - 1) // (backend_conf.ngpus_current_node if backend_conf.ngpus_current_node > 0 and backend_conf.distributed else 1))
    dataloaders = []
    for n, ds in datasets.items():
        shuffle = True if n == 'trainset' else False
        batch_size = hp['batch_size'] if n == 'trainset' else hp['batch_size'] * 32  # NOTE: Evaluation batches are choosen to be 32 times larger than train batches
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=not backend_conf.is_cuda)
        # Setup data batch prefetching by patching dataloader(s) if asked so
        dataloaders.append(dataloader_prefetch_batches(dl, device=backend_conf.device) if hp['prefetch_batches'] else dl)
    trainset, *validset_testset = dataloaders

    model = model.to(device, non_blocking=True)
    model = _setup_distributed_training(device, backend_conf, model, (trainset.batch_size, *trainset.dataset[0][0].shape), use_sync_batch_norm=hp['use_sync_batch_norm'])
    losses = _setup_ignite_losses(losses=losses, loss_weights=loss_weights, device=device)
    optimizer = opt(model.parameters(), **hp['optimizer_opts'])
    output_path = add_training_output_dir(hp['output_path'], backend_conf)
    scheduler = None
    if hp['scheduler'] is not None:
        args_to_eval = hp['scheduler']['eval_args'] if 'eval_args' in hp['scheduler'] else {}
        scheduler_kwargs = {n: eval(v, {'hp': hp, 'iterations': len(trainset)}) if n in args_to_eval else v
                            for n, v in hp['scheduler']['kwargs'].items()}
        scheduler = hp['scheduler']['type'](optimizer=optimizer, **scheduler_kwargs)

    def process_function(engine: Engine, batch: Tuple[TENSOR_OR_SEQ_OF_TENSORS_T]) -> Dict[str, float]:
        nonlocal hp, device, model, losses, loss_weights, optimizer
        if hp['prefetch_batches'] and hasattr(engine._dataloader_iter, '_prefetched_batch'):
            # Don't need to move batches to device memory: Batch comes from a dataloader patched with `dataloader_prefetch_batches` which prefetches batches to device memory duing computation
            x, *y = batch
        else:
            if hp['prefetch_batches']:
                # TODO: remove this message once debugged
                logging.warn('Warning: Batch prefetching not enabled even if `prefetch_batches` is `True` (non-cuda device? or code error)')
            x, *y = tuple(ignite.utils.convert_tensor(b, device=device, non_blocking=True) for b in batch)
        if len(y) == 1:
            y = y[0] # Only one target tensor

        # Apply model on input batch(es) `x`
        model.train()
        y_pred = model(x)

        # Evaluate loss(es)/metric(s) function(s) and train model by performing a backward prop followed by an optimizer step
        batch_losses = {n: loss(y_pred, y) for n, loss in losses.items()}
        optimizer.zero_grad()
        batch_losses[MAIN_TRAINING_LOSS_NAME].backward()
        optimizer.step()
        return {n: loss.item() for n, loss in batch_losses.items()}

    trainer = Engine(process_function)

    # TODO: Figure out if sampler handling is done right and test distributed training further
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset) if backend_conf.distributed else None
    if backend_conf.distributed and not callable(getattr(train_sampler, 'set_epoch', None)):
        raise ValueError(f'Error: `trainset` DataLoader\'s sampler should have a method `set_epoch` (train_sampler=`{train_sampler}`)')
    to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer} + (dict() if scheduler is None else {'scheduler': scheduler})
    metric_names = ['_batch_train_loss', *losses.keys()]
    common.setup_common_training_handlers(trainer,
                                          train_sampler=train_sampler,
                                          to_save=to_save,
                                          save_every_iters=hp['save_every_iters'],
                                          output_path=str(output_path),
                                          lr_scheduler=scheduler,
                                          with_gpu_stats=True,
                                          output_names=metric_names,
                                          with_pbars=True,
                                          with_pbar_on_iters=True,
                                          log_every_iters=hp['log_progress_every_iters'],
                                          device=backend_conf.device)

    if backend_conf.rank == 0:
        tb_logger = TensorboardLogger(log_dir=str(output_path))
        tb_logger.attach(trainer, log_handler=OutputHandler(tag='train', metric_names=metric_names), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer, param_name='lr'), event_name=Events.ITERATION_STARTED)
        # TODO: make sure hp params logging works here + use test eval metrics instead of training's
        tb_logger.attach(trainer, log_handler=HyperparamsOutoutHandler(hp, metric_names=metric_names), event_name=Events.COMPLETED)

    def _metrics(prefix): return {**{f'{prefix}_{n}': m for n, m in metrics.items()},
                                  **{f'{prefix}_{n}': loss for n, loss in losses.items()}}

    valid_evaluator = create_supervised_evaluator(model, metrics=_metrics('valid'), device=device, non_blocking=True)
    train_evaluator = create_supervised_evaluator(model, metrics=_metrics('train'), device=device, non_blocking=True)

    @trainer.on(Events.EPOCH_STARTED(every=hp['validate_every_epochs']))
    @trainer.on(Events.COMPLETED)
    def _run_validation(engine: Engine):
        if torch.cuda.is_available() and not backend_conf.is_cpu:
            torch.cuda.synchronize()

        # Trainset evaluation
        train_state = train_evaluator.run(trainset)
        train_metrics = {f'train_{n}': float(v) for n, v in train_state.metrics.items()}
        for n, v in train_metrics.items():
            mlflow.log_metric(n, v, step=engine.state.epoch)

        # Validset evaluation
        valid_state = valid_evaluator.run(validset_testset[0])
        valid_metrics = {f'valid_{n}': float(v) for n, v in valid_state.metrics.items()}
        for n, v in valid_metrics.items():
            mlflow.log_metric(n, v, step=engine.state.epoch)

        if not is_nni_run_standalone():
            # TODO: make sure `valid_state.metrics` is ordered so that reported default metric to NNI is always the same
            nni.report_intermediate_result({'default': valid_state.metrics.values()[0], **train_metrics, **valid_metrics})

    if backend_conf.rank == 0:
        event = Events.ITERATION_COMPLETED(every=hp['log_progress_every_iters'] if hp['log_progress_every_iters'] else None)
        ProgressBar(persist=False, desc='Train evaluation').attach(train_evaluator, event_name=event)
        ProgressBar(persist=False, desc='Test evaluation').attach(valid_evaluator)

        log_handler = OutputHandler(tag='train', metric_names=list(metrics.keys()), global_step_transform=global_step_from_engine(trainer))
        tb_logger.attach(train_evaluator, log_handler=log_handler, event_name=Events.COMPLETED)

        log_handler = OutputHandler(tag='test', metric_names=list(metrics.keys()), global_step_transform=global_step_from_engine(trainer))
        tb_logger.attach(valid_evaluator, log_handler=log_handler, event_name=Events.COMPLETED)

        # Store the best model by validation accuracy:
        common.save_best_model_by_val_score(str(output_path), valid_evaluator, model=model, metric_name='accuracy', n_saved=3, trainer=trainer, tag='val')

        if hp['log_grads_every_iters'] is not None and hp['log_grads_every_iters'] > 0:
            tb_logger.attach(trainer, log_handler=GradsHistHandler(model, tag=model.__class__.__name__), event_name=Events.ITERATION_COMPLETED(every=hp['log_grads_every_iters']))

    if hp['crash_iteration'] is not None and hp['crash_iteration'] >= 0:
        @trainer.on(Events.ITERATION_STARTED(once=hp['crash_iteration']))
        def _(engine):
            raise Exception('STOP at iteration: {}'.format(engine.state.iteration))

    if nni_compression_pruner is not None:
        # Notify NNI compressor (pruning or quantization) for each epoch and eventually each steps/batch-iteration if need by provided Pruner/Quantizer (see NNI Compression Documentation for more details: https://nni.readthedocs.io/en/latest/Compressor/QuickStart.html#apis-for-updating-fine-tuning-status)
        @trainer.on(Events.EPOCH_STARTED)
        def _nni_compression_update_epoch(engine):
            nni_compression_pruner.update_epoch(engine.state.epoch)

        if getattr(nni_compression_pruner, 'step', None) is Callable:
            @trainer.on(Events.ITERATION_COMPLETED)
            def _nni_compression_batch_step(engine):
                nni_compression_pruner.step()

    _resume_training(hp.get('resume_from'), to_save)

    try:
        logging.info(f'> ignite runs training loop for "{model}" model...')
        state = trainer.run(trainset, max_epochs=hp['epochs'])
        logging.info(f'Training of "{model}" model done sucessfully.')

        if not is_nni_run_standalone():
            # Report final training results to NNI (NNI HP or NNI Classic NAS APIs)
            # TODO: make sure `valid_state.metrics` is ordered so that reported default metric to NNI is always the same
            nni.report_final_result({'default': valid_evaluator.state.metrics.values()[0], **train_evaluator.state.metrics, **valid_evaluator.state.metrics})
        return (valid_evaluator.state.metrics, state)
    except Exception as e:
        logging.error(
            f'Ignite training loop of "{type(model).__name__}" model failed, exception "{e}" raised{deepcv.utils.NL}### Traceback ###{deepcv.utils.NL}{traceback.format_exc()}')
        raise RuntimeError(f'Error: `{e}` exception raised during ignite training loop of "{type(model).__name__}" model...') from e
    finally:
        if backend_conf.rank == 0:
            tb_logger.close()
        if hp['log_output_dir_to_mlflow'] and mlflow.active_run():
            logging.info('Logging training output directory as mlfow artifacts...')
            mlflow.log_artifacts(str(output_path))
            # TODO: log and replace artifacts to mlflow at every epochs?
            # TODO: make sure all artifacts are loaded synchronously here
            # shutil.rmtree(output_path)


def _setup_distributed_training(device, backend_conf: BackendConfig, model: torch.nn.Module, batch_shape: torch.Size, use_sync_batch_norm: bool = False) -> torch.nn.Module:
    if backend_conf.distributed:
        # Setup distributed training with `torch.distributed`
        dist.init_process_group(backend_conf.dist_backend, init_method=backend_conf.dist_url)
        assert backend_conf.is_cuda, 'Error: Distributed training must be run on GPU(s).'
        torch.cuda.set_device(backend_conf.device)
        # TODO: make sure we dont want to add more device IDs here (see distributed examples in Ignite or PyTorch)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[backend_conf.local_rank, ], output_device=backend_conf.local_rank)

        if use_sync_batch_norm and any(map(model.modules(), lambda m: isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)))):
            # Convert batch normalization sub-modules to `SyncBatchNorm` before wraping model with DDP (DistributedDataParallel), allowing to synchronizes statistics across nodes/GPUs in distributed setups, which can be usefull when batch size is too small on a single node/GPU
            # NOTE: `convert_sync_batchnorm` is needed as direct usage of `SyncBatchNorm` doesnt upports DDP with mutliple GPU per process according to PyTorch 1.6.0 documentation
            # TODO: may have been fixed since muti GPU per process use case of DDP have been fixed in 1.6.0 release?
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif not backend_conf.is_cpu and is_data_parallelization_usefull_heuristic(model, batch_shape):
        # If not distributed, we can still use data parrallelization if there are multiple GPUs available and data is large enought to be worth it
        model = data_parallelize(model)
    return model


def _resume_training(resume_from: Union[str, Path], to_save: Dict[str, Any]):
    if resume_from:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), f'Checkpoint "{checkpoint_fp}" is not found'
        print(f'Resuming from a checkpoint: {checkpoint_fp}')
        checkpoint = torch.load(checkpoint_fp.as_posix())
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()

# TODO: remove this deprecated code and use ignite training loop instead

# def eval_loop(model: torch.nn.Module, loss: torch.nn.modules.loss._Loss, eval_dataloader: DataLoader):
#     yield train_eval_loop(model, loss, eval_dataloader)


"""
    best_valid_loss, best_train_loss = float("inf"), float("inf")
    best_run_epoch = -1
    epochs_since_best_loss = 0

    # Main training loop
    for epoch in range(1, epochs + 1):
        print(f"{deepcv.utils.NL}Epoch %03d/%03d{deepcv.utils.NL}" % (epoch, epochs) + '-' * 15)
        train_loss = 0

        trange, update_bar = tu.progress_bar(trainset, '> Training on trainset', min(
            len(trainset.dataset), trainset.batch_size), custom_vars=True, disable=not pbar)
        for (batch_x, colors, bbs) in trange:
            batch_x, colors, bbs = batch_x.to(DEVICE).requires_grad_(True), tu.flatten_batch(colors.to(DEVICE)), tu.flatten_batch(bbs.to(DEVICE))

            def closure():
                optimizer.zero_grad()
                output_colors, output_bbs = model(batch_x)
                loss = bce_loss_scale * pos_metric(output_colors, colors) + bb_metric(output_bbs, bbs)
                loss.backward()
                return loss
            loss = float(optimizer.step(closure).clone().detach())
            scheduler.step()
            train_loss += loss / len(trainset)
            update_bar(trainLoss=f'{len(trainset) * train_loss / (trange.n + 1):.7f}', lr=f'{float(scheduler.get_lr()[0]):.3E}')

        print(f'>\tDone: TRAIN_LOSS = {train_loss:.7f}')
        valid_loss = evaluate(epoch, model, validset, bce_loss_scale, best_valid_loss, pbar=pbar)
        print(f'>\tDone: VALID_LOSS = {valid_loss:.7f}')
        if best_valid_loss > valid_loss:
            print('>\tBest valid_loss found so far, saving model...')  # TODO: save model
            best_valid_loss, best_train_loss = valid_loss, train_loss
            best_run_epoch = epoch
            epochs_since_best_loss = 0
        else:
            epochs_since_best_loss += 1
            if early_stopping is not None and early_stopping > 0 and epochs_since_best_loss >= early_stopping:
                print(f'>\tModel not improving: Ran {epochs_since_best_loss} training epochs without improvement. Early stopping training loop...')
                break

    print(f'>\tBest training results obtained at {best_run_epoch}nth epoch (best_valid_loss = {best_valid_loss:.7f}, best_train_loss = {best_train_loss:.7f}).')
    return best_train_loss, best_valid_loss, best_run_epoch


def evaluate(epoch: int, model: torch.nn.Module, validset: DataLoader, bce_loss_scale: float, best_valid_loss: float, pbar: bool = True) -> float:
    model.eval()
    with torch.no_grad():
        bb_metric, pos_metric = torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss()
        valid_loss = 0.

        for step, (batch_x, colors, bbs) in enumerate(tu.progress_bar(validset, '> Evaluation on validset', min(len(validset.dataset), validset.batch_size), disable=not pbar)):
            batch_x, colors, bbs = batch_x.to(DEVICE), tu.flatten_batch(colors.to(DEVICE)), tu.flatten_batch(bbs.to(DEVICE))
            output_colors, output_bbs = model(batch_x)
            valid_loss += (bce_loss_scale * pos_metric(output_colors, colors) + bb_metric(output_bbs, bbs)) / len(validset)

            if step >= len(validset) - 1 and VIS_DIR is not None and best_valid_loss >= valid_loss:
                print(f"> ! Saving visualization images of inference on some validset values...")
                for idx in np.random.permutation(range(validset.batch_size))[:8]:
                    img, bbs, _cols = datasets.retrieve_data(batch_x[idx], output_bbs[idx], output_colors[idx])
                    vis.show_bboxes(img, bbs, datasets.COLORS, out_fn=VIS_DIR / f'vis_valid_{idx}.png')
    return float(valid_loss)
"""

# loss
# metrics = {'mse': , }
# ToSave = TypedDict('Savable', {'model': torch.nn.Module, 'optimizer': Optional[Optimizer], 'scheduler': Optional[_LRScheduler], 'epoch': int})

# class Metrics(Generic[T]):

#     Sequence[ignite.metrics.Metric]


# Metrics = TypedDict('Metrics', {str: ignite.metrics.Metric, ...})
# for name, metric in metrics.items():
#     metric.compute()
#     if isinstance(metric, ignite.metrics.Loss):

# #train_metrics, valid_metrics: namedtuple('Metrics', ['loss']),  namedtuple('Metrics', ['loss'])

# with SummaryWriter(log_dir='', comment="LR_0.1_BATCH_16") as writer:
#     for epoch, metrics in train_eval_loop(denoiser, loss, validset, trainset, optimizer, scheduler, summary=writer):
#         print(f'TRAIN_LOSS={metrics.train_loss}')
#         print(f'VALID_LOSS={metrics.valid_loss}')


# def train_eval_loop(epochs : int, model: torch.nn.Module, loss: torch.nn.modules.loss._Loss, validset: DataLoader, trainset: DataLoader = None, optimizer: Optimizer = None,
#                     scheduler: _LRScheduler = None, custom_metrics: Metrics = None, summary: SummaryWriter = None, device: torch.device = get_device(), disable_progress_bar: bool = False):

#     train = optimizer is not None and scheduler is not None and trainset is not None
#     if (optimizer is not None or scheduler is not None or trainset is not None) and not train:
#         logging.error('ERROR: Incoherent arguments passed to train_eval_loop: optimizer, scheduler and trainset_dataloader are either all or none specified (training + eval or eval only)')
#     elif not DataLoader.drop_last:
#         logging.error('ERROR: This traininig/evaluation loop only support dataloader with drop_last==True (running metrics averages would be false otherwise due to last batch size)')
#     else:

#         # Training/evaluation loop
#         def _epoch(is_training):
#             dataloader = trainset if is_training else validset
#             batches, update_bar = progress_bar(dataloader, '> Training' if is_training else '> Evaluation', disable=disable_progress_bar)
#             for step, (inputs, targets) in enumerate(batches):
#                 inputs = inputs.to(device)
#                 targets = targets.to(device)
#                 optimizer.zero_grad()
#                 mean_loss = 0.

#                 # Initialize eval metrics
#                 if not is_training:
#                     for name, metric in metrics:
#                         metric.reset()

#                 with torch.set_grad_enabled(is_training):
#                     # Forward and evaluate loss
#                     outputs = model(inputs)
#                     batch_loss = loss(outputs, targets)

#                     if is_training:
#                         # Backward propagate gradients and optimize (if training)
#                         batch_loss.backward()
#                         optimizer.step()
#                         scheduler.step()
#                     else:
#                         # Update eval metrics
#                         for name, metric in metrics:
#                             if isinstance(metric, ignite.metrics.Loss) or isinstance(metric, ignite.metrics.LambdaLoss):
#                             metric.update(batch_loss.item())


#                 # Update running metric averages (Here we asume dataloader.batch_size == inputs.size(0) because DataLoader.drop_last == True)
#                 # TODO: update progress bar with custom metrics
#                 mean_loss = mean_loss * (1. - 1. / step) + ensure_mean_batch_loss(loss, batch_loss, sum_divider=inputs.size(0)).item() / step
#                 batches.set_postfix(step=step, batch_size=inputs.size(0), mean_loss=f'{mean_loss:.4E}', lr=f'{scheduler.get_lr().item():.3E}')
#                 return mean_loss

#         # Train and evaluate model
#         # TODO: yield custom metrics here
#         for epoch in range(epochs):
#             custom_metrics = custom_metrics if custom_metrics is not None else {}
#             metrics = Namespace(**{'valid_loss': float('inf'), 'train_loss': float('inf')}.update(custom_metrics))
#             if train:
#                 model.train()
#                 metrics.train_loss = _epoch(True)
#             model.eval()
#             metrics.valid_loss = _epoch(False)

#             for name, metric in metrics:
#                 value = metric.compute()
#                 summary.add_scalar(name, value, global_step=epoch)

#             for name, value in vars(metrics) if isinstance(value, deepcv.utils.NUMBER_T):
#                 summary.add_scalar(name, value, global_step=epoch)
#             yield epoch, metrics
