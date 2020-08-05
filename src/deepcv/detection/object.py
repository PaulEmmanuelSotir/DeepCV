#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Object detection module - object.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Any, Dict, Optional, Tuple, Callable, List, Iterable, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import ignite
from ignite.metrics import Accuracy
from kedro.pipeline import Pipeline, node

import deepcv.utils
import deepcv.meta.nn
import deepcv.meta.base_module
import deepcv.meta.nni_tools
import deepcv.meta.hyperparams
import deepcv.meta.data.preprocess
import deepcv.meta.ignite_training
from deepcv.meta.data.datasets import batch_prefetch_dataloader_patch
from deepcv.meta.types_aliases import *

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'


# TODO: Object detector model ...


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
