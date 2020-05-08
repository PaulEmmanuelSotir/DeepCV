#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Portability meta module - portability.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn

from tests.tests_utils import test_module_cli

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

if __name__ == '__main__':
    cli = test_module_cli(__file__)
    cli()

# TODO: Use onnyx and onnyx-runtime for inference?: https://pytorch.org/docs/stable/onnx.html
