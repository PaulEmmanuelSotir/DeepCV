#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Uncertainty estimation meta module - uncertainty.estimation.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
    - TODO: see https://ai.googleblog.com/2020/01/can-you-trust-your-models-uncertainty.html for a comparision of different uncertainty estimation methods (github: https://github.com/google-research/google-research/tree/master/uq_benchmark_2019)
    - TODO: Look into https://github.com/google-research/google-research/tree/master/uncertainties, especially in order to dig more into MCMC SGD aproaches.
"""
import torch
import torch.nn as nn

from deepcv import utils


__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

if __name__ == '__main__':
    cli = utils.import_tests().test_module_cli(__file__)
    cli()
