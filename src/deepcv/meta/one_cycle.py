#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Object detection module - object.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Optional, Callable, Iterable, Union, Tuple

from tests.tests_utils import test_module_cli

# TODO: allow to determine optimal weight decay during one cycle policy 'hyperparameter search' (see https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6)
# TODO: see https://arxiv.org/pdf/1803.09820.pdf and https://arxiv.org/pdf/1506.01186.pdf
# TODO: investigate combination of one cycle policy with different learning rate for each neural net layers
# TODO: make sure one cycle policy can be applied in distributed configuration too
# TODO: inherit from a scheduler class and/or integrate with ignite training handlers

__all__ = ['OneCyclePolicy', 'run_param_search', 'find_optimal_params', 'plot_search_curves']


class OneCyclePolicy:
    def __init__(self, base_lr: float = 1e-4, max_lr: float = 1e9, base_momentum: Optional[float] = None, max_momentum: Optional[float] = None):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum

    def next_lr(self) -> Union[Tuple(float, float), float]:
        raise NotImplementedError


PARAMS_SRARCH_RSLT_T = Union[List[float], List[Tuple[float, float]]]


def run_param_search(loss, training_process: Callable, momentum_search: bool = True, weigth_decay_search: Optional[Iterable[float]]) -> PARAMS_SRARCH_RSLT_T:
    raise NotImplementedError


def find_optimal_params(params_search_rslts: PARAMS_SRARCH_RSLT_T):
    raise NotImplementedError


def plot_search_curves(params_search_rslts: PARAMS_SRARCH_RSLT_T):
    raise NotImplementedError


if __name__ == '__main__':
    cli = test_module_cli(__file__)
    cli()
