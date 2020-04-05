#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Hyperparameter search meta module - hpsearch.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np
from typing import Sequence, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...tests.tests_utils import test_module

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

# TODO: implement tools for NNI (https://github.com/microsoft/nni) usage (NNI Board and NNICTL) + MLFlow versionning and viz

# TODO: + make use of CONSTRUCTIVE PREDICTION OF THE GENERALIZATION ERROR ACROSS SCALES (https://openreview.net/pdf?id=ryenvpEKDr) in order to predict model accuracy for given hyperparameter set from training on small dataset subsets


def get_dataset_subsets(data: DataLoader, subsets: Sequence[float]) -> Iterable[DataLoader]


if __name__ == '__main__':
    test_module(__file__)
