#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Quantization and distillation meta module - quantistillation.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn

from ...tests.tests_utils import test_module

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

# TODO: implement tools for compression, distillation and quantization using NNI's compression tools (see https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/compression) and/or pytorch compression/quantization (see https://pytorch.org/docs/master/quantization.html#introduction-to-quantization)

if __name__ == '__main__':
    test_module(__file__)
