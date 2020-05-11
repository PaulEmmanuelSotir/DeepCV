#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Portability meta module - portability.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn

from deepcv import utils


__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

if __name__ == '__main__':
    cli = utils.import_tests().test_module_cli(__file__)
    cli()

# TODO: Use onnyx and onnyx-runtime for inference?: https://pytorch.org/docs/stable/onnx.html
