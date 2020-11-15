#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Contrastive learning meta module - contrastive.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
    - TODO: investigate triplet network training as a special case of distillation (include this tooling in distillation framework?)
    - TODO: implement various contrastive learning tooling/losses
    - TODO: Look into Mutual Knowledge Distillation (= bidirectional KD = DML(Deep-Mutual-Learning)) where teacher and student networks both learn from each other.  
        'Dense Cross-layer Mutual-distillation' (DCM) paper makes usage of 'Deep Supervision' to improve over existing DML methods by sharing knwoledge mutually both at output layer level and at hidden layers level.
        DCM thus appends, to regular DML, mutual knowledge distiallation to hidden features representations thanks to Deep-Supervision methods): [Knowledge Transfer via Dense Cross-layer Mutual-distillation](https://github.com/sundw2014/DCM), paper: https://arxiv.org/pdf/2008.07816v1.pdf
    - TODO: Also look into 'Deep supervision' itself. The basic idea is to train temporary extra classifiers/NN-Models on intermediate features for better convergence, alternative inference tasks and two-way/mutual knowledge transfer/distilation (those extra models are droped once model is trained)
"""
import functools
from typing import Tuple, Union, Sequence, Optional

import numpy as np

import torch
import torch.nn
import torch.nn.functional
from torch.utils.data import DataLoader

import deepcv.utils
from .nn import func_to_module


__all__ = ['jensen_shannon_divergence_consistency_loss', 'sample_triplets', 'JensenShannonDivergenceConsistencyLoss', 'TripletMarginLoss']
__author__ = 'Paul-Emmanuel Sotir'


def jensen_shannon_divergence_consistency_loss(net: torch.nn.Module, original: torch.Tensor, *augmented_n: Sequence[torch.Tensor], reduction: str = 'batchmean', to_log_probabilities: bool = True):
    """ Functionnal Implementation of Jensen Shannon Divergence Consistency Loss as defined in [AugMix DeepMind's paper](https://arxiv.org/pdf/1912.02781.pdf).
    Args:
        - to_log_probabilities: If `net` already outputs a distribution in log-propabilities (e.g. logsoftmax output layer), set `to_log_probabilities` to `False`, otherwise, if `net` outputs are regular probabilities, let it to `True`: Underlying 'torch.nn.functional.kl_div' needs input distribution to be log-probabilities and applies a log operator to target distribution
    """
    kl_div = functools.partial(torch.nn.functional.kl_div, reduction=reduction, log_target=not to_log_probabilities)
    with torch.no_grad():
        # Avoid unescessary back prop through NN applied to original image (as first adviced in [Virtual Adversarial Training 2018 paper](https://arxiv.org/pdf/1704.03976.pdf), or [UDA consistency loss (2019)](https://arxiv.org/pdf/1904.12848.pdf))
        p_original = net(original)
    p_augmented_n = [net(aug_n) for aug_n in augmented_n]
    M = torch.mean(torch.stack([p_original, *p_augmented_n]), dim=0)
    if to_log_probabilities:
        M = M.log()
    else:
        # TODO: remove these exp operators and use new pytorch 1.6 kl_div parameter 'log_target' (set to `not to_log_probabilities`)
        p_original, p_augmented_n = p_original.exp(), [p.exp() for p in p_augmented_n]
    return torch.mean(torch.stack([kl_div(M, p_original), *[kl_div(M, p_n) for p_n in p_augmented_n]]), dim=0)


def sample_triplets(dataset: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def label_smoothing_xentropy_loss(smooth_labels: bool = False, smoothing_eps: FLOAT_OR_FLOAT_TENSOR_T = 0.1):
    """ Appends label smoothing regumarization support to `torch.functional.cross_entropy` cross-entropy loss (which combines Log-Softmax and Negative-Log-Likelihood Loss).  
    Label smoothing regularization implementation is inspired from ["attention is all you need" paper implementation from Google](https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py#L38).  
    Args:
        - smooth_labels: If `False`, this function just returns `torch.functional.cross_entropy`. If `True`, returned callable processes cross-entropy loss with label smoothing regularization.
        - smoothing_eps: Epsilon probability to be used for label smoothing: If there are `C` classes, hard target probabilities of `1` and `0` will be respectively 'smoothed' to `1 - smoothing_eps` and `smoothing_eps / (C - 1)`.
    Returns callable to be used as functional cross-entropy loss implementation (with or without label smoothing regularization depending on given `smooth_labels` boolean)
    """
    if not smooth_labels:
        # Regular cross entropy without label smoothing regularization
        return torch.functional.cross_entropy
    
    def _xentropy_label_smoothing_forward(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None, ignore_index: Optional[int] = -100, reduction: str = 'mean') -> torch.FloatTensor:
        target = target.contiguous().view(-1) # Flattened view of target tensor
        n_class = pred.size(1) if not multiclass_labels else 0.5
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - smoothing_eps) + smoothing_eps / (n_class - 1.) # TODO: figure out why there is no ' - 1.' in original code ?!
        return torch.functional.cross_entropy(...) # TODO:..

    def _xentropy_label_smoothing_forward(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None, ignore_index: Optional[int] = -100, reduction: str = 'mean') -> torch.FloatTensor:
        target = target.contiguous().view(-1) # Flattened view of target tensor
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - smoothing_eps) + (1 - one_hot) * smoothing_eps / (n_class - 1) # Apply label smoothing probabilities to one_hot targets
        log_prb = torch.functional.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        if weight is not None:
            # Deduce weights for batch samples based on their target label and apply those weights on cross entropy loss
            weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)
            loss = torch.mul(loss, weight)
        if ignore_index is not None:
            non_pad_mask = target.ne(ignore_index)
            loss = loss.masked_select(non_pad_mask)
        # Apply reduction function on cross entropy loss and return it
        return loss.sum()
    return _xentropy_label_smoothing_forward

# NOTE: When training a model with triplet margin loss, try out to enable 'swap' option (swaps anchor and positive if distance between negative and positive is lower than distance between anchor and negative)
TripletMarginLoss = torch.nn.TripletMarginLoss
JensenShannonDivergenceConsistencyLoss = func_to_module(jensen_shannon_divergence_consistency_loss, init_params=['net', 'reduction', 'to_log_probabilities'])


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
