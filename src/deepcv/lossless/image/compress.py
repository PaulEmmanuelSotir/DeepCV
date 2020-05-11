#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Image lossless compression module - compress.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn

from deepcv import utils


__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

# TODO: Implement custom lossless compression based on DropBox's Lepton (improvement over lossless JPEG algorithm) modified to include 2 shallow neural net heuristics for better wavelets and brightness predictions (thus lower file size)

if __name__ == '__main__':
    cli = utils.import_tests().test_module_cli(__file__)
    cli()
