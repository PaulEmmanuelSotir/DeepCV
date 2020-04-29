#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Multitask training utilities meta module - multitask.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn

from ...tests.tests_utils import test_module

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

"""
meta.multitask module for training utilities in a multitask setup, where a model learns to resolve multiple unsupervised or supervised tasks at once (à la [MAML](https://arxiv.org/pdf/1908.10400.pdf) & co)
TODO: SOTA Review of multitask learning
TODO: specific tools in case of self supervised multitask learning (probably most interesting use case in DeepCV)
"""

if __name__ == '__main__':
    test_module(__file__)
