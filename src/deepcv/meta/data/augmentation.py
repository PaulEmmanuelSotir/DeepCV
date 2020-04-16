#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Data augmentation meta module - augmentation.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import numpy as np

import torch
import torch.nn as nn

from ....tests.tests_utils import test_module

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'


# TODO: parse YAML parameters for augmentations reciepes
# TODO: implement various augmentation operators: sharpness, crop, brightness, contrast, tweak_colors, gamma, noise, rotate, translate, scale, smooth_non_linear_deformation
# TODO: implement augmentation based on distilled SinGAN model
# TODO: AugMix augmentation recipe implementation? (see https://arxiv.org/pdf/1912.02781.pdf and parameters.yml)

#### AugMix pseudo-code: ####
# 1: Input: Model pˆ, Classification Loss L, Image xorig, Operations O = {rotate, . . . , posterize}
# 2: function AugmentAndMix(xorig, k = 3, α = 1)
# 3: Fill xaug with zeros
# 4: Sample mixing weights (w1, w2, . . . , wk) ∼ Dirichlet(α, α, . . . , α)
# 5: for i = 1, . . . , k do
# 6: Sample operations op1
# , op2
# , op3 ∼ O
# 7: Compose operations with varying depth op12 = op2
# ◦ op1
# and op123 = op3
# ◦ op2
# ◦ op1
# 8: Sample uniformly from one of these operations chain ∼ {op1
# , op12, op123}
# 9: xaug += wi
# · chain(xorig) . Addition is elementwise
# 10: end for
# 11: Sample weight m ∼ Beta(α, α)
# 12: Interpolate with rule xaugmix = mxorig + (1 − m)xaug
# 13: return xaugmix
# 14: end function
# 15: xaugmix1 = AugmentAndMix(xorig) . xaugmix1 is stochastically generated
# 16: xaugmix2 = AugmentAndMix(xorig) . xaugmix1 6= xaugmix2
# 17: Loss Output: L(ˆp(y | xorig), y) + λ Jensen-Shannon(ˆp(y | xorig); ˆp(y|xaugmix1); ˆp(y|xaugmix2))


if __name__ == '__main__':
    test_module(__file__)
