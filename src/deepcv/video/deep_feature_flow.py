#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Deep Feature Flow module - deep_feature_flow.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir
"""
from typing import Tuple, Sequence

import torch
from torch import nn

import deepcv.utils

def deep_feature_flow_inference(video_frames: Iterable[torch.Tensor], features_model: nn.Module, tasks_models: Sequence[nn.Module], flow_estimation: nn.Module, keyframe_period: int) -> Generator[Dict[int, Tuple], None, None]:
    """ Implementation of [Deep Feature Flow for Video Recognition 2017 Paper](https://arxiv.org/abs/1611.07715)
    Args:
        - video_frames: Generator or iterable which yields input video frames
        - features_model:
        - tasks_models:
        - keyframe_period: Frame period at which model is performing inference without propagation. Inference on frames in between which are not key-frames will be performed on propagated features using optical flow (avoids applying `features_model` for each video frames). """
    assert keyframe_period > 1, f'Error: Keyframe period must be greater than 1: `keyframe_period={keyframe_period}`.'
    feature_maps = features_model(video_frames.__next__())
    yield {0: (task_network(feature_maps) for task_network in tasks_models)}
    
    def _scale(current_frame: torch.Tensor, key_frame: torch.Tensor) -> torch.Tensor:
        return ... # TODO: implement _scale function
    
    def _propagation_W(keyframe_features: torch.Tensor, 2D_flow_field: torch.Tensor, scale_field: torch.Tensor) -> torch.Tensor:
        return ... # TODO: implement _propagation_W function

    for i, frame in enumerate(video_frames):
        if (i + 1) % keyframe_period == 0:
            feature_maps = features_model(frame)
            keyframe, keyframe_features = frame, feature_maps
        else:
            # Propagation using optical flow and previous key frame results
            scale_field = _scale(frame, keyframe) 
            2D_flow_field = flow_estimation(frame, keyframe)
            feature_maps = _propagation_W(keyframe_features, 2D_flow_field, scale_field) 
        yield {i + 1: (task_network(feature_maps) for task_network in tasks_models)}

if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
