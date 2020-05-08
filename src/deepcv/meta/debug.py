#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Debugging meta module - debug.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn

from...tests.tests_utils import test_module

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

if __name__ == '__main__':
    cli = test_module_cli(__file__)
    cli()

# TODO: use https://pytorch.org/docs/stable/bottleneck.html, see https://docs.python.org/3/library/profile.html
# TODO: Generate per block or per layer computationnal costs: like in CSP paper: https://arxiv.org/pdf/1911.11929v1.pdf
