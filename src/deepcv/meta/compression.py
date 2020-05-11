#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Model quantization, distillation and pruning meta module - compression.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
	- TODO: implement tools for compression, distillation and quantization using NNI's compression tools (see https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/compression) and/or pytorch compression/quantization (see https://pytorch.org/docs/master/quantization.html#introduction-to-quantization)
	- TODO: Perform quantization and distillation along with optional network pruning (e.g. convolution channel/filter pruning using Rigl or more likely "knapsack-pruning")

.. See pruning methods in [Paper with code leaderboard for Network pruning](https://paperswithcode.com/task/network-pruning)
4 approaches which seems the closest to SOTA:
	- [knapsack-pruning-with-inner-distillation deepai paper](https://arxiv.org/pdf/2002.08258v1.pdf)
	- [TAS: Network pruning via Transformable Architecture Search](https://arxiv.org/pdf/1905.09717v5.pdf)
	- [DRAWING EARLY-BIRD TICKETS: TOWARDS MORE EFFICIENT TRAINING OF DEEP NETWORKS](https://openreview.net/pdf?id=BJxsrgStvr)
	- [FPGM: Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/pdf/1811.00250.pdf)
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
