#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Regularization meta module - regularization.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn

import deepcv.utils as utils
test_module_cli = utils.import_tests().test_module_cli

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'


def get_regularization_term(y, ypred, y_tranform=None, ypred_tranform=None):
    raise NotImplementedError


if __name__ == '__main__':
    cli = test_module_cli(__file__)
    cli()
