#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Self supervised training meta module - self_supervised.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from collections import OrderedDict
from types import SimpleNamespace
from typing import Iterable
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import deepcv.utils
from deepcv.meta.types_aliases import *


__all__ = []
__author__ = 'Paul-Emmanuel Sotir'


class Task(ABC):
    def __init__(self, hp: SimpleNamespace): self.hp = hp
    def __repr__(self) -> str: return f'class="{self.__class__.__name__}";hp="{self.hp}"'

    @abstractmethod
    def loss(self) -> LOSS_FN_T: ...
    @abstractmethod
    def head_model(self, embeding_shape: torch.Size) -> nn.Module: ...
    @abstractmethod
    def sample_data(self, dl: DataLoader) -> Iterable[torch.Tensor]: ...


def TrainOnTasks(embdding: nn.Module, tasks: Iterable[Task], embedding_shape: torch.Size):
    """ Self supervised inference heads
    Pytorch module which appends to an embedding model, multiple siamese inference heads in order to solve different kinds of self supervised tasks.
    """
    raise NotImplementedError
    losses = [t.loss() for t in tasks]
    heads_modules = [(f'task_{t.__repr__()}', t.head_model(embedding_shape)) for t in tasks]


class TasksDataSampler():
    """ Self supervised tasks data sampler
    Samples and preprocess data from given image dataloader according to each self supervised tasks to be solved.
    TODO: Inherit from torch sampler or so?
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        raise NotImplementedError

    def sample(self, count: int = 1):
        raise NotImplementedError


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
