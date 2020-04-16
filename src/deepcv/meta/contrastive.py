#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Contrastive learning meta module - contrastive.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import functools
from typing import Tuple, Union

import numpy as np

import torch
import torch.nn as nn
from torch.functional import F
from torch.utils.data import DataLoader

from deepcv import utils
from deepcv.meta.nn import func_to_module
from ...tests.tests_utils import test_module

__all__ = ['jensen_shannon_divergence_consistency_loss', 'sample_triplets', 'JensenShannonDivergenceConsistencyLoss', 'TripletMarginLoss']
__author__ = 'Paul-Emmanuel Sotir'

# TODO: investigate triplet network training as a special case of distillation (include this tooling in distillation frameork?)
# TODO: implement various contrastive learning tooling/losses


def jensen_shannon_divergence_consistency_loss(net: nn.Module, original: torch.Tensor, *augmented_n, reduction: str = 'batchmean', log_target: bool = False):
    """ Functionnal Implementation of Jensen Shannon Divergence Consistency Loss as defined in [AugMix DeepMind's paper](https://arxiv.org/pdf/1912.02781.pdf). """
    kl_div = functools.partial(F.kl_div, reduction=reduction)
    p_original = net(original)
    p_augmented_n = [net(aug_n) for aug_n in augmented_n]
    M = torch.mean(torch.stack([p_original, *p_augmented_n], dim=0), dim=0)
    return torch.mean(torch.stack([kl_div(p_original, M), *[kl_div(p_n, M) for p_n in p_augmented_n]]), dim=0)


def sample_triplets(dataset: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError


JensenShannonDivergenceConsistencyLoss = func_to_module(jensen_shannon_divergence_consistency_loss, init_params=['net', 'reduction', 'log_target'])
TripletMarginLoss = nn.TripletMarginLoss

if __name__ == '__main__':
    test_module(__file__)
