#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Distributed inference and training meta module - distributed.py - `DeepCV`__
Merges all embeddings from 'embeddings' submodule into one embedding.
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn

import deepcv.utils as utils


__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

# TODO: use pytorch and/or Apex distribution tools...

if __name__ == '__main__':
    cli = utils.import_tests().test_module_cli(__file__)
    cli()
