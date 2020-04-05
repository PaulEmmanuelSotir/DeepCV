#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Distributed inference and training meta module - distributed.py - `DeepCV`__
Merges all embeddings from 'embeddings' submodule into one embedding.
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn

from ...tests.tests_utils import test_module

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

# TODO: use pytorch and/or Apex distribution tools...

if __name__ == '__main__':
    test_module(__file__)
