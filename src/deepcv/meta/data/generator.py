
#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Data generator meta module - generator.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import types
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

import click
from click import secho, style

from deepcv import utils
from ....tests.tests_utils import test_module

singan = utils.try_import(r'....third_party.SinGAN.SinGAN', msg='Can\'t import third party module')


__all__ = ['DistilledSinGAN', 'train_distilled_singan']
__author__ = 'Paul-Emmanuel Sotir'


# Use SinGAN: Learning a Generative Model from a Single Natural Image https://arxiv.org/abs/1905.01164  ; SinGAN github: https://github.com/tamarott/SinGAN


class DistilledSinGAN(nn.Module):
    def __init__(self):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def train_distilled_singan(train_images):
    if singan:
        # We first train a new SinGAN instance on each images
        for img in train_images:
            raise NotImplementedError
            singan_opts = types.SimpleNamespace(device=utils.get_device(), mode='train', input_dir=r'', input_name=r'')
            Gs, Zs, reals, NoiseAmp = [], [], [], []

            real = singan.functions.read_image(singan_opts)
            singan.functions.adjust_scales2image(real, singan_opts)
            singan.training.train(singan_opts, Gs, Zs, reals, NoiseAmp)
            singan.manipulate.SinGAN_generate(Gs, Zs, reals, NoiseAmp, singan_opts)


if __name__ == '__main__':
    test_module(__file__)
