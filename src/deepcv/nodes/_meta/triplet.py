#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Triplet network training meta module - triplet.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Tuple, Union
import numpy as np

import torch
import torch.nn as nn
from torch.functional import F
from torch.utils.data import DataLoader

from deepcv import utils
from ....tests.tests_utils import test_module

__all__ = ['sample_triplets', 'triplet_loss']
__author__ = 'Paul-Emmanuel Sotir'

# TODO: implement triplet network loss and associated tools
# TODO: investigate triplet network training as a special case of distillation (include this tooling in distillation frameork?)


def sample_triplets(dataset: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def triplet_loss(net: nn.Module, anchor: torch.tensor, positive: torch.tensor, negative: torch.tensor, margin: Union[float, torch.Tensor] = 0.1, norm_order: torch.Number = 2):
    """
    Function evaluating neural net's triplet loss on given anchor, positive and negative batches/inputs

    Args:
        net: Pytorch module on which loss is evaluated
        anchor: Anchor input image(s) somewhat similar to positive image(s)
        positive: Positive input image(s) somewhat similar to anchor image(s)
        negative: Negative input image(s) disimilar to anchor and positive image(s)
        margin: Margin scalar added to triplet loss before torch.max operator (also often called alpha in triplet loss litterature)
        norm_order: order of distance metric used in triplet loss, passed to ``p`` argument of :func:`torch.dist`
    """
    isscalar = isinstance(margin, utils.Number) or (isinstance(margin, torch.Tensor) and margin.dim() == 0)
    assert isscalar, f'Error: Invalid argument "margin" passed to triplet_loss function (margin={margin}).'
    a, p, n = net(anchor), net(positive), net(negative)
    return F.relu(torch.dist(a, p, p=norm_order) - torch.dist(a, n, p=norm_order) + margin)


if __name__ == '__main__':
    test_module(__file__)
