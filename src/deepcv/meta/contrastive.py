#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Contrastive learning meta module - contrastive.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
    - TODO: investigate triplet network training as a special case of distillation (include this tooling in distillation framework?)
    - TODO: implement various contrastive learning tooling/losses
"""
import functools
from typing import Tuple, Union, Sequence

import numpy as np

import torch
import torch.nn as nn
from torch.functional import F
from torch.utils.data import DataLoader

import deepcv.utils
import deepcv.meta.nn


__all__ = ['jensen_shannon_divergence_consistency_loss', 'sample_triplets', 'JensenShannonDivergenceConsistencyLoss', 'TripletMarginLoss']
__author__ = 'Paul-Emmanuel Sotir'


def jensen_shannon_divergence_consistency_loss(net: nn.Module, original: torch.Tensor, *augmented_n: Sequence[torch.Tensor], reduction: str = 'batchmean', to_log_probabilities: bool = True):
    """ Functionnal Implementation of Jensen Shannon Divergence Consistency Loss as defined in [AugMix DeepMind's paper](https://arxiv.org/pdf/1912.02781.pdf).
    Args:
        - to_log_probabilities: If `net` already outputs a distribution in log-propabilities (e.g. logsoftmax output layer), set `to_log_probabilities` to `False`, otherwise, if `net` outputs are regular probabilities, let it to `True`: Underlying 'torch.nn.functional.kl_div' needs input distribution to be log-probabilities and applies a log operator to target distribution
    """
    kl_div = functools.partial(F.kl_div, reduction=reduction, log_target=not to_log_probabilities)
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


# NOTE: When training a model with triplet margin loss, try out to enable 'swap' option (swaps anchor and positive if distance between negative and positive is lower than distance between anchor and negative)
TripletMarginLoss = nn.TripletMarginLoss
JensenShannonDivergenceConsistencyLoss = deepcv.meta.nn.func_to_module(jensen_shannon_divergence_consistency_loss, init_params=['net', 'reduction', 'to_log_probabilities'])

if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
