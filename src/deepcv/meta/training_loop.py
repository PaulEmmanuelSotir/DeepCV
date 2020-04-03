#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Training loop meta module - training_loop.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn
import ignite

from ....tests.tests_utils import test_module
from deepcv import utils

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'


def init_training(deterministic: bool = True):
    if deterministic:
        utils.set_seeds(353453)
    utils.setup_cudnn(deterministic)
    device = utils.get_device()
    raise NotImplementedError  # TODO: implement


if __name__ == '__main__':
    test_module(__file__)


# TODO: remove this deprecated code and use ignite training loop instead

# def eval_loop(model: nn.Module, loss: nn.loss._Loss, eval_dataloader: DataLoader):
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
# for name, metric in metrics:
#     metric.compute()
#     if isinstance(metric, ignite.metrics.Loss):

# #train_metrics, valid_metrics: namedtuple('Metrics', ['loss']),  namedtuple('Metrics', ['loss'])

# with SummaryWriter(log_dir='', comment="LR_0.1_BATCH_16") as writer:
#     for epoch, metrics in train_eval_loop(denoiser, loss, validset, trainset, optimizer, scheduler, summary=writer):
#         print(f'TRAIN_LOSS={metrics.train_loss}')
#         print(f'VALID_LOSS={metrics.valid_loss}')


# def train_eval_loop(epochs : int, model: nn.Module, loss: nn.loss._Loss, validset: DataLoader, trainset: Optional[DataLoader] = None, optimizer: Optional[Optimizer] = None,
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
