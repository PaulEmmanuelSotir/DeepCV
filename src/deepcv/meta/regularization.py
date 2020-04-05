#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Regularization meta module - regularization.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn

from ...tests.tests_utils import test_module

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'


def get_regularization_term(y, ypred, y_tranform=None, ypred_tranform=None):
    raise NotImplementedError


if __name__ == '__main__':
    test_module(__file__)
