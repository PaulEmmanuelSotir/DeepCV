
# -*- coding: utf-8 -*-
""" Generative model learning utilities meta module - generative.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
    - TODO: basic GAN and VAE model definition, training and insight tooling
    - TODO: Video/image generation quality metric: https://github.com/google-research/google-research/tree/master/frechet_video_distance from this paper: https://arxiv.org/pdf/1812.01717.pdf (could give ideas like using VAE embedding to make sure Pr and Pg are gaussian multivariate distribution and use 2-Wassterstein/frechet metric: d(PR, PG) = minX,Y E|X − Y |^2 = |µR − µG|^2 + Tr (ΣR + ΣG − 2sqrt(ΣRΣG) if Pr and Pg ~ MultivariateGaussian)
"""
import torch
import torch.nn as nn

from deepcv import utils


__all__ = []
__author__ = 'Paul-Emmanuel Sotir'


if __name__ == '__main__':
    cli = utils.import_tests().test_module_cli(__file__)
    cli()
