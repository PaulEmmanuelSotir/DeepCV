#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Video Optical Flow estimation module - optical_flow.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import torch
from torch import nn

import deepcv.utils

__all__ = ['FlowNetV2']
__author__ = 'Paul-Emmanuel Sotir'

class FlowNetV2(nn.Module):
    # TODO: append FlowNetV2 implementation here
    def __init__(self):
        super(FlowNetV2, self).__init__()
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return args[0]
        
if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
