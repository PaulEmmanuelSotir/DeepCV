#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Training loop meta module - training_loop.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import logging
import traceback
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Iterable, Tuple, Union, Type

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

import ignite
from ignite.utils import convert_tensor
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Loss, Metric
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.contrib.engines import common
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler, GradsHistHandler
import ignite.contrib.handlers

import deepcv.utils
import deepcv.meta.nn
import deepcv.meta.hyperparams


__all__ = ['BackendConfig', 'train']
__author__ = 'Paul-Emmanuel Sotir'


class BackendConfig:
    def __init__(self, device=None, dist_backend: dist.Backend = None, dist_url: str = '', local_rank: Optional[int] = 0):
        self.device = deepcv.utils.get_device(devid=local_rank) if device is None else device
        self.is_cpu = self.device.type == 'cpu'
        self.ncpu = multiprocessing.cpu_count()
        self.dist_backend = dist_backend
        self.dist_url = dist_url
        self.distributed = dist_backend is not None and dist_backend != ''
        self.local_rank = local_rank
        self.ngpus_current_node = torch.cuda.device_count()
        self.rank, self.ngpus, self.nnodes = 0, 0, 1

        if self.distributed:
            self.rank = dist.get_rank()
            self.ngpus = dist.get_world_size()
            self.nnodes = dist.get_world_size() // self.ngpus_current_node  # TODO: fix issue: self.nnodes correct only if each nodes have the same GPU count
        else:
            self.ngpus = self.ngpus_current_node

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if self.is_cpu:
            return f'single-node-cpu-{self.ncpu}'
        if self.ngpus <= 1:
            return 'single-gpu'
        return f'distributed-{self.nnodes}nodes-{self.ngpus}gpus-{self.ngpus_current_node}current-node-gpus'


def train(hp: Union[deepcv.meta.hyperparams.Hyperparameters, Dict[str, Any]], model: nn.Module, loss: nn.modules.loss._Loss, dataloaders: Tuple[DataLoader], opt: Type[torch.optim.Optimizer], backend_conf: BackendConfig = BackendConfig(), metrics: Dict[str, Metric] = {}) -> ignite.engine.State:
    """ Pytorch model training procedure defined using ignite
    Args:
        - hp: Hyperparameter dict, see ```deepcv.meta.ignite_training._check_params`` to see required and default training (hyper)parameters
        - model: Pytorch ``nn.Module`` to train
        - loss: Loss module to be used
        - dataloaders: Tuple of pytorch DataLoader giving access to trainset, validset and an eventual testset
        - opt: Optimizer type to be used for gradient descent
        - backend_conf: Backend information defining distributed configuration (available GPUs, whether if CPU or GPU are used, distributed node count, ...), see ``deepcv.meta.ignite_training.BackendConfig`` class for more details.
        - metrics: Additional metrics dictionnary (loss is already included in metrics to be evaluated by default)
    Returns a [`ignite.engine.state`](https://pytorch.org/ignite/engine.html#ignite.engine.State) object which describe ignite training engine's state (iteration, epoch, dataloader, max_epochs, metrics, ...).
    # TODO: print training initialization info
    # TODO: add support for cross-validation
    # TODO: Report training metrics and results to MLFlow?
    """
    TRAINING_HP_DEFAULTS = {'output_path': ..., 'optimizer_opts': ..., 'shceduler': ..., 'epochs': ...,
                            'validate_every': 1, 'checkpoint_every': 1000, 'log_model_grads_every': -1,
                            'display_iters': 1000, 'seed': None, 'deterministic': False, 'resume_from': '', 'crash_iteration': -1}
    logging.info(f'Starting ignite training procedure to train "{model}" model...')
    assert len(dataloaders) == 3 or len(dataloaders) == 2, 'Error: dataloaders tuple must either contain: `trainset and validset` or `trainset, validset and testset`'
    hp, _ = deepcv.meta.hyperparams.to_hyperparameters(hp, TRAINING_HP_DEFAULTS, raise_if_missing=True)
    output_path = Path(hp['output_path'])
    trainset, *validset_testset = dataloaders
    device = backend_conf.device

    if hp['deterministic']:
        deepcv.utils.set_seeds(backend_conf.rank + hp['seed'])
    deepcv.utils.setup_cudnn(deterministic=hp['deterministic'])

    model = model.to(device)
    model = _setup_distributed_training(device, backend_conf, model, trainset[0])

    if backend_conf.local_rank == 0 and backend_conf.rank == 0:
        # Create output directory if current node is master or if not distributed
        now = datetime.now().strftime(r'%Y%m%d-%H%M%S')
        output_path = Path(hp['output_path']) / f'{now}-{backend_conf}'
        if not output_path.exists():
            output_path.mkdir(parents=True)

    loss = loss.to(device)
    optimizer = opt(model.parameters(), **hp['optimizer_opts'])
    schedule = hp['scheduler']
    scheduler = schedule['type'](**{n: eval(v) if 'eval_args' in schedule and n in schedule['eval_args'] else v for n, v in schedule['kwargs'].items()})

    def process_function(engine, batch):
        x, y = (convert_tensor(b, device=device, non_blocking=True) for b in batch)

        model.train()
        # Supervised part
        y_pred = model(x)
        loss_tensor = loss(y_pred, y)

        optimizer.zero_grad()
        loss_tensor.backward()
        optimizer.step()
        return {'batch loss': loss_tensor.item(), }

    trainer = Engine(process_function)
    # TODO: figure out why 'None' if not distributed?
    # TODO: replace it with torch.utils.data.distributed.DistributedSampler(trainset) ?
    train_sampler = trainset.sampler if backend_conf.distributed else None
    to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer, 'scheduler': scheduler}
    metric_names = ['batch loss', ]
    common.setup_common_training_handlers(trainer,
                                          train_sampler=train_sampler,
                                          to_save=to_save,
                                          save_every_iters=hp['checkpoint_every'],
                                          output_path=hp['output_path'],
                                          lr_scheduler=scheduler,
                                          output_names=metric_names,
                                          with_pbar_on_iters=hp['display_iters'],
                                          log_every_iters=10,)

    if backend_conf.rank == 0:
        tb_logger = TensorboardLogger(log_dir=hp['output_path'])
        tb_logger.attach(trainer, log_handler=OutputHandler(tag='train', metric_names=metric_names), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer, param_name='lr'), event_name=Events.ITERATION_STARTED)

    metrics = {**metrics, 'loss': Loss(loss, device=device if backend_conf.distributed else None)}

    valid_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    def run_validation(engine):
        torch.cuda.synchronize()
        train_evaluator.run(trainset)
        valid_evaluator.run(validset_testset[0])

    trainer.add_event_handler(Events.EPOCH_STARTED(every=hp['validate_every']), run_validation)
    trainer.add_event_handler(Events.COMPLETED, run_validation)

    if backend_conf.rank == 0:
        if hp['display_iters']:
            ProgressBar(persist=False, desc='Train evaluation').attach(train_evaluator)
            ProgressBar(persist=False, desc='Test evaluation').attach(valid_evaluator)

        log_handler = OutputHandler(tag='train', metric_names=list(metrics.keys()), global_step_transform=global_step_from_engine(trainer))
        tb_logger.attach(train_evaluator, log_handler=log_handler, event_name=Events.COMPLETED)

        log_handler = OutputHandler(tag='test', metric_names=list(metrics.keys()), global_step_transform=global_step_from_engine(trainer))
        tb_logger.attach(valid_evaluator, log_handler=log_handler, event_name=Events.COMPLETED)

        # Store the best model by validation accuracy:
        common.save_best_model_by_val_score(hp['output_path'], valid_evaluator, model=model, metric_name='accuracy', n_saved=3, trainer=trainer, tag='val')

        if hp['log_model_grads_every'] is not None and hp['log_model_grads_every'] > 0:
            tb_logger.attach(trainer, log_handler=GradsHistHandler(model, tag=model.__class__.__name__),
                             event_name=Events.ITERATION_COMPLETED(every=hp['log_model_grads_every']))

    if hp['crash_iteration'] is not None and hp['crash_iteration'] >= 0:
        @trainer.on(Events.ITERATION_STARTED(once=hp['crash_iteration']))
        def _(engine):
            raise Exception('STOP at iteration: {}'.format(engine.state.iteration))

    _resume_training(hp.get('resume_from'), to_save)

    try:
        logging.info(f'> ignite runs training loop for "{model}" model...')
        state = trainer.run(trainset, max_epochs=hp['epochs'])
        logging.info(f'Ignite training procedure of "{model}" model sucessfully done.')
        return state
    except Exception as e:
        logging.error(f'Ignite training loop of "{model}" model failed, exception "{e}" raised\n### Traceback ###\n{traceback.format_exc()}')
        raise RuntimeError(f'Error: Error occured during ignite training loop of "{model}" model...') from e
    finally:
        if backend_conf.rank == 0:

            tb_logger.close()


def _setup_distributed_training(device, backend_conf: BackendConfig, model: nn.Module, batch_shape: torch.Size) -> nn.Module:
    if backend_conf.distributed:
        dist.init_process_group(backend_conf.dist_backend, init_method=backend_conf.dist_url)
        assert device.type == 'cuda', 'Error: Distributed training must be run on GPU(s).'
        torch.cuda.set_device(backend_conf.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[backend_conf.local_rank, ], output_device=backend_conf.local_rank)
    elif not backend_conf.is_cpu and deepcv.meta.nn.is_data_parallelization_usefull_heuristic(model, batch_shape):
        # If not distributed, we can still use data parrallelization if there are multiple GPUs available and data is large enought to be worth it
        model = deepcv.meta.nn.data_parallelize(model)
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

# def eval_loop(model: nn.Module, loss: torch.nn.modules.loss._Loss, eval_dataloader: DataLoader):
#     yield train_eval_loop(model, loss, eval_dataloader)


"""
    best_valid_loss, best_train_loss = float("inf"), float("inf")
    best_run_epoch = -1
    epochs_since_best_loss = 0

    # Main training loop
    for epoch in range(1, epochs + 1):
        print("\nEpoch %03d/%03d\n" % (epoch, epochs) + '-' * 15)
        train_loss = 0

        trange, update_bar = tu.progess_bar(trainset, '> Training on trainset', min(
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


def evaluate(epoch: int, model: nn.Module, validset: DataLoader, bce_loss_scale: float, best_valid_loss: float, pbar: bool = True) -> float:
    model.eval()
    with torch.no_grad():
        bb_metric, pos_metric = torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss()
        valid_loss = 0.

        for step, (batch_x, colors, bbs) in enumerate(tu.progess_bar(validset, '> Evaluation on validset', min(len(validset.dataset), validset.batch_size), disable=not pbar)):
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
# ToSave = TypedDict('Savable', {'model': nn.Module, 'optimizer': Optional[Optimizer], 'scheduler': Optional[_LRScheduler], 'epoch': int})

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


# def train_eval_loop(epochs : int, model: nn.Module, loss: torch.nn.modules.loss._Loss, validset: DataLoader, trainset: Optional[DataLoader] = None, optimizer: Optional[Optimizer] = None,
#                     scheduler: Optional[_LRScheduler] = None, custom_metrics: Optional[Metrics] = None, summary: SummaryWriter = None, device: torch.device = get_device(), disable_progress_bar: bool = False):

#     train = optimizer is not None and scheduler is not None and trainset is not None
#     if (optimizer is not None or scheduler is not None or trainset is not None) and not train:
#         logging.error('ERROR: Incoherent arguments passed to train_eval_loop: optimizer, scheduler and trainset_dataloader are either all or none specified (training + eval or eval only)')
#     elif not DataLoader.drop_last:
#         logging.error('ERROR: This traininig/evaluation loop only support dataloader with drop_last==True (running metrics averages would be false otherwise due to last batch size)')
#     else:

#         # Training/evaluation loop
#         def _epoch(is_training):
#             dataloader = trainset if is_training else validset
#             batches, update_bar = progess_bar(dataloader, '> Training' if is_training else '> Evaluation', disable=disable_progress_bar)
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
#                 mean_loss = mean_loss * (1. - 1. / step) + mean_batch_loss(loss, batch_loss, inputs.size(0)) / step
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

#             for name, value in vars(metrics) if isinstance(value, Number):
#                 summary.add_scalar(name, value, global_step=epoch)
#             yield epoch, metrics
