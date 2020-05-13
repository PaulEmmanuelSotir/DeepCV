#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Image/Video feature keypoints detection module - keypoints.detector.py - `DeepCV`__  
Implements image/video keypoint detector from [detector part of Unsupervised Learning of Object Structure and Dynamics from Videos](https://arxiv.org/pdf/1906.07889.pdf), [Google Research's official implentation](https://github.com/google-research/google-research/tree/master/video_structure) is based on Tensorflow, thus, we had to implement it ourselves.  
.. moduleauthor:: Paul-Emmanuel Sotir  

*TODO List*  
    - TODO: Video interpolation / dynamics learning and also relevant for unsupervised keypoint detection ideas: https://github.com/google-research/google-research/tree/master/video_structure from this paper: https://arxiv.org/abs/1906.07889   
    - TODO: NN model for Keypoints proposal from a conv NN which outputs K feature maps: each output channel is normalized and averaged into (x,y) coordinates in order to obtain relevant keypoints (K at most). Trained using a autoencoder setup: a generator (decoder with end-to-end skip connection from anchor frame) must be able to reconstruct input image from keypoints (converted to gaussian heat maps) and another frame along with its own keypoints (e.g. first video frame)  
    - TODO: Modify keypoint model in order to have feature pattern information associated with keypoint coordinates (instead of simply associate input image)?  
"""
import logging

import torch
from torch import nn

import deepcv.utils

__all__ = ['KeypointsDetector']
__author__ = 'Paul-Emmanuel Sotir'


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
