#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Uncertainty estimation meta module - uncertainty.estimation.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
    - TODO: see https://ai.googleblog.com/2020/01/can-you-trust-your-models-uncertainty.html for a comparision of different uncertainty estimation methods (github: https://github.com/google-research/google-research/tree/master/uq_benchmark_2019)
    - TODO: Look into https://github.com/google-research/google-research/tree/master/uncertainties, especially in order to dig more into MCMC SGD aproaches.
    - TODO: use torch.distributions for helper/tooling function making easier to infer prob distribution (like Mixture Density Networks). In some cases, it is prefered to infer prob densities and uncertainty over parameters of infered distribution instead of directly infering value(s) and uncertainty over value(s)
"""
import deepcv.utils
from ..types_aliases import *


__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
