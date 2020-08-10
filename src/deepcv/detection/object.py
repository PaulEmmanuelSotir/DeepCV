#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Object detection module - object.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Any, Dict, Optional, Tuple, Callable, List, Iterable, Union

import torch
import torch.nn
import torch.optim
from torch.utils.data import Dataset

import ignite
from ignite.metrics import Accuracy
from kedro.pipeline import Pipeline, node

import deepcv.utils
import deepcv.meta.data.preprocess
from deepcv.meta.types_aliases import HYPERPARAMS_T, METRICS_DICT_T

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'


# TODO: Object detector model ...


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
