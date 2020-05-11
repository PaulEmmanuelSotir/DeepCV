#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Video lossless compression module - compress.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn

from deepcv import utils


__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

# TODO: before custom implementation, try to use AV1 codec or VP9 codec

if __name__ == '__main__':
    cli = utils.import_tests().test_module_cli(__file__)
    cli()
