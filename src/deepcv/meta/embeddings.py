#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Embeddings meta module - embedddings.py - `DeepCV`__
Merges all embeddings from 'embeddings' submodule into one embedding.
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
