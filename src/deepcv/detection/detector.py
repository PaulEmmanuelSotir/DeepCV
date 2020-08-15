#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Object detection module - detector.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import logging
from typing import Any, Dict, Optional, Tuple, Callable, List, Iterable, Union

import torch
import torch.nn
import torch.optim
from torch.utils.data import Dataset

import ignite
from ignite.metrics import Accuracy
from kedro.pipeline import Pipeline, node

import deepcv.utils
import deepcv.meta
from deepcv.meta.types_aliases import HYPERPARAMS_T, METRICS_DICT_T
import deepcv.meta.data.preprocess

__all__ = ['get_pipelines']
__author__ = 'Paul-Emmanuel Sotir'


# TODO: Object detector model ...

def get_pipelines() -> Dict[str, Pipeline]:
    return dict()


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
