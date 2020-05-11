#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Training loop meta module - training_loop.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Any

import torch
import torch.nn as nn

from deepcv import utils


__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

# TODO: Create a code base dataset from github (filtered on open licenses)
# TODO: add pretrained transformer language models (e.g BERT, ...) and eventually fine tune them on code dataset
# TODO: implement deep reinforcement learning framework from existing SOTA project or gym project
# TODO: use reinforcement learning framework/environement to train custom models on top of language understanding models ... bla bla
# TODO: append/combine to this project simple common tools like linting, snippets, classical static_analysis, stylegyding (e.g. convert name from camel case to underscore case and vis versa)
# TODO: setup online learning from developer/framework-specific model specialization/training


class analyser(nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, *input: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError


if __name__ == '__main__':
    cli = utils.import_tests().test_module_cli(__file__)
    cli()
