#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Data visualization meta module - viz.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import collections
from pathlib import Path
from types import ModuleType
from typing import Optional, Union, Tuple

import numpy as np
import matplotlib

import torch
from PIL import Image

import deepcv.utils


__all__ = ['plot', 'sample_tumbails', 'plot_tumbails']
__author__ = 'Paul-Emmanuel Sotir'


def plot(backend: ModuleType = matplotlib, show_in_tensorboard: bool = False, **plot_kwargs):
    raise NotImplementedError
    # if backend is matplotlib:
    #     img = matplotlib.plot(**plot_kwargs)
    # elif backend is seaborn:
    #     img = seaborn.plot(**plot_kwargs)
    # elif backend is altair:
    #     img = altair.plot(**plot_kwargs)
    # else:
    #     raise ValueError(f'Error: Unrecognized plotting backend: "{backend}"')

    if show_in_tensorboard:
        raise NotImplementedError


def sample_tumbails(datatset, image_count: int = 64, tumbail_size: Tuple[int, int] = (32, 32)):
    with datatset.threadlocal_shuffle():
        for image, _taget in datatset[:image_count]:
            tumb = image.thumbnail(tumbail_size)
            raise NotImplementedError  # TODO: aggregate tumbails into one PIL Image


def plot_tumbails(tumbails):
    raise NotImplementedError

# TODO: generate (or at least simplify generation) architectecture diagram by feeding pytorch computation graph into http://alexlenail.me/NN-SVG/index.html


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
