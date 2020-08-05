#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" One Cycle scheduling policy meta module - one_cycle.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Optional, Callable, Iterable, Union, Tuple, List

import deepcv.utils


# TODO: allow to determine optimal weight decay during one cycle policy 'hyperparameter search' (see https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6)
# TODO: see https://arxiv.org/pdf/1803.09820.pdf and https://arxiv.org/pdf/1506.01186.pdf
# TODO: investigate combination of one cycle policy with different learning rate for each neural net layers
# TODO: make sure one cycle policy can be applied in distributed configuration too
# TODO: inherit from a scheduler class and/or integrate with ignite training handlers

__all__ = ['OneCyclePolicy', 'run_param_search', 'find_optimal_params', 'plot_search_curves']
__author__ = 'Paul-Emmanuel Sotir'


class OneCyclePolicy:
    def __init__(self, base_lr: float = 1e-4, max_lr: float = 1e9, base_momentum: Optional[float] = None, max_momentum: Optional[float] = None):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum

    def next_lr(self) -> Union[Tuple[float, float], float]:
        raise NotImplementedError


PARAMS_SRARCH_RSLT_T = Union[List[float], List[Tuple[float, float]]]


def run_param_search(training_process: Callable, momentum_search: bool = True, weigth_decay_search: Optional[Tuple[float, float]] = None) -> PARAMS_SRARCH_RSLT_T:
    """ Runs a quick learning rate search, by training and evaluating model on a few iterations and deduce optimal lr/momentum/decay according to an heuristic/thumb-rule.
    This function is quite similar to fastai's [`lr_finder`](https://docs.fast.ai/callbacks.lr_finder.html), but also allows you to search for optimal momentum and/or weight decay. 
    Args:
        - training_process: Callable which run one training iteration and evaluates model (should return valid loss and take learning rate and eventually momentum and/or weight decay)
        - momentum_search: Boolean indicating whether if momentum should also be search for, along with learning rate. (will then return optimal momentum along with optimal learning rate)
        - weigth_decay_search: Optional list of weight decays to search for. If not None, then hyperparameter search will also look for optimal weight decay value among those provided
    """
    raise NotImplementedError


def find_optimal_params(params_search_rslts: PARAMS_SRARCH_RSLT_T):
    raise NotImplementedError


def plot_search_curves(params_search_rslts: PARAMS_SRARCH_RSLT_T):
    raise NotImplementedError


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
