#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Data visualization meta module - viz.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import collections
import numpy as np
from pathlib import Path
from typing import Optionnal, Union, Tuple

import torch
import torch.nn as nn
from PIL import Image

from .datatset import Image_t, ImageDataset
from ....tests.tests_utils import test_module

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'


def sample_tumbails(self, datatset, image_count=64, tumbail_size: Tuple[int, int] = (32, 32)):
    with datatset.threadlocal_shuffle():
        for image, _taget in datatset[:image_count]:
            tumb = image.thumbnail(tumbail_size)
            raise NotImplementedError  # TODO: aggregate tumbails into one PIL Image

# TODO: generate (or at least simplify generation) architectecture diagram by feeding pytorch computation graph into http://alexlenail.me/NN-SVG/index.html


if __name__ == '__main__':
    test_module(__file__)
