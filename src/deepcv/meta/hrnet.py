#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" HRNet sub-modules - hrnet.py - `DeepCV`__
Defines various [High Resolution neural Network (HRNet)](https://arxiv.org/abs/1908.07919) and [Pyramidal Convolution (PyConv)](https://arxiv.org/pdf/2006.11538.pdf) NN architecture building blocks like `ParallelConvolution`, `MultiresolutionFusion`, `HRNetInputStem`
.. moduleauthor:: Paul-Emmanuel Sotir

*To-Do List*
    - TODO: ...
"""
import inspect
import logging
from collections import OrderedDict
from typing import Callable, Optional, Type, Union, Tuple, Iterable, Dict, Any, Sequence, List, Hashable, Mapping

import numpy as np

import torch
import torch.nn
import torch.nn.functional

import deepcv.utils
from deepcv.utils import NL
from .nn import conv_nd, layer
from .types_aliases import *

__all__ = ['ParallelConvolution', 'MultiresolutionFusion', 'HRNetV1RepresentationHead', 'HRNetV2RepresentationHead', 'HRNetV2pRepresentationHead', 'HRNetInputStem']
__author__ = 'Paul-Emmanuel Sotir'


class ParallelConvolution(torch.nn.Module):
    """ParallelConvolution multi-resolution/parallel convolution module which is a generalization of multi-resolution convolution modules of [HRNet paper](https://arxiv.org/abs/1908.07919) and Pyramidal Convolutions as described in [PyConv paper](https://arxiv.org/pdf/2006.11538.pdf).
    This `torch.nn.Module` operation performs a grouped conv operation where each groups are regular/independant convolutions which may be applied on different feature map resolutions (like HRNet parallel streams/branches) and/or have different conv parameters, like different kernel sizes for each group (like PyConv does).
    Thus, this module can be interpreted as being multiple regular convolutions applied to different feature maps and performed in parrallel NN branches.
    NOTE: Chaining such grouped multi-res convs in a `torch.nn.Sequential` is a simple way to define a siamese NN of parrallel convolutions. Moreover, combined with `deepcv.meta.nn.MultiresolutionFusion` module, `ParallelConvolution` allows to define HRNet-like NN architecture, where information from each siamese branches of parallel multi-res convs can flow between those throught fusion modules. (Multi-Resolution fusion modules can be compared to 'non-grouped' multi-resolution convolution, see repsective doc and HRNet paper for more details)
    # TODO: Asuchronously yield each output for each branch?
    """

    def __init__(self, input_shape: Union[torch.Size, Sequence[torch.Size]], submodule_params: Dict[str, Tuple[Sequence[Any], Any]], act_fn: Union[Sequence[Optional[Type]], Optional[Type]] = torch.nn.ReLU,
                 preactivation: Union[Sequence[bool], bool] = False, dropout_prob: float = None, norm_type: Sequence[NormTechnique] = None, norm_kwargs: Sequence[Dict[str, Any]] = None,
                 supported_norm_ops: NORM_TECHNIQUES_MODULES_T = None, channel_dim: int = 1, raise_on_not_enough_in_tensors: bool = True):
        """ Multi-resolution/parallel convolution module which is a generalization of multi-resolution convolution modules of [HRNet paper](https://arxiv.org/abs/1908.07919) and Pyramidal Convolutions as described in [PyConv paper](https://arxiv.org/pdf/2006.11538.pdf).
        This `torch.nn.Module` operation performs a grouped conv operation where each groups are regular/independant convolutions which may be applied on different feature map resolutions (like HRNet parallel streams/branches) and/or have different conv parameters, like different kernel sizes for each group (like PyConv does).
        Args:
            - input_shape: Input tensors shape(s), with channel dim located at `channel_dim`th dim and with minibatch dimension. These tensor shapes should be shapes on which each parrallel/multi-res convolutions will be performed. This argument can also be used by eventual normalization technique(s) (`input_shape[channel_dim:]` passed to `deepcv.meta.nn.normalization_techniques`).
            - submodule_params: Keyword arguments dict passed `deepcv.meta.nn.conv_nd`. Each argument can either be a sequence (List, Tuple, ...) of the same lenght as `input_shape` or a single value depending on if each multires-group convs have differents arg values or share the same argument value.
            - ... for other arguments, see corresponding arguments in `deepcv.meta.nn.layer` documentation.
                NOTE: Note that `act_fn` and `preactivation`, like arguments in `submodule_params`, can also be sequences if different values for each group/parallel convs are needed, but normalization parameters and `dropout_prob` will be shared across all group/parallel convs (no support for different `dropout_prob`, `norm_type`, `norm_kwargs` and `supported_norm_ops` args across parallel convs for now)
            - channel_dim: Channel dim index in provided `input_shape` tensor shapes. Dimensions after channel dim should be spatial features dims on which convolution is applied (e.g. two dimensions after channel dim for a 2D convoltion)
            - raise_on_not_enough_in_tensors: Boolean indicating whether to raise an exception if too few input tensors are given during forward passes (less than `len(input_shape)` input tensors given to forward method)

        NOTE: `kernel_size`, `padding` and `dilation` args in `submodule_params` must allways be specified as a sequence of integers (one for each dims, e.g. `kernel_size` cant be a single int, like `2` for a `2x2` kernel, otherwise, there could be ambiguities as conv args can be specified once or as a sequence of values for each parrallel conv groups/branches). I.e., their value must match this type: `Union[Sequence[Sequence[int]], Sequence[int]]` where inner-most sequence being of lenght `dims` and optional parent sequence being of the smae lenght as `input_shape` sequence. See also related error message in function code.
        NOTE: This is an helper function which actually is a special use case of `deepcv.meta.nn.layer` function; For more details about `deepcv.meta.nn.parallel_convolution` arguments, see `deepcv.meta.nn.layer` documentation.

        .. See also related `deepcv.meta.nn.MultiresolutionFusion` module which is also part of basic HRNet architecture building blocks.
        .. See also [`HigherHRNet` paper](https://arxiv.org/pdf/1908.10357.pdf).
        """
        if isinstance(input_shape, torch.Size):
            input_shape = [input_shape, ]
        elif len(input_shape) == 0:
            raise ValueError('Error in `deepcv.meta.nn.parallel_convolution`, `input_shape`(s) argument must be a sequence of `torch.Size` with at least one input tensor shape(s). '
                             f'NOTE: If you are performing a single conv on a single tensor shape/resolution, use `deepcv.meta.nn.layer` instead. Got `input_shape="{input_shape}"`')
        self.raise_on_not_enough_in_tensors = raise_on_not_enough_in_tensors
        self.channel_dim = channel_dim
        self.spatial_dims = len(input_shape[0][channel_dim+1:])

        # Share the same activation function and/or preactivation for all group/parallel convolution layers if those are not sequences
        if not isinstance(act_fn, Sequence):
            act_fn = [act_fn, ] * len(input_shape)
        if not isinstance(preactivation, Sequence):
            preactivation = [preactivation, ] * len(input_shape)
        # If act_fn and/or preactivation where sequences, those should have the same lenght as `input_shape` (one different value for each parallel convolution)
        if any([not len(input_shape) == len(seq) for seq in {act_fn, preactivation}]):
            raise ValueError('Error: If `preactivation` and/or `act_fn` args are sequences, those should have the same lenght as `input_shape` (one value for each parallel convolution)')

        # Forbid to specify a single int instead of a sequence of `dims` ints. Nescessary to avoid ambiguities between a sequence of different args for each group/parallel convs or a sequence of ints for each feature map dim (would be ambiguous when `dims==len(input_shape)`)
        conv_args_with_sequence_constraint = {'padding', 'kernel_size', 'dilation'}

        # Handle convs parameters to allow to either specify different parameters for each resolution(s)/group or a single parameter common to all mulit-res group convs.
        all_conv_ops_kwargs = list()
        for i in range(len(input_shape)):
            ith_conv_kwargs = dict(dims=self.spatial_dims)
            for n, v in submodule_params.items():
                if n in conv_args_with_sequence_constraint:
                    if isinstance(v, Sequence) and len(v) == len(input_shape) and all([isinstance(sub_v, Sequence) for sub_v in v]):
                        v = v[i]
                    elif not isinstance(v, Sequence) or len(v) not in {1, self.spatial_dims} or any([isinstance(sub_v, Sequence) for sub_v in v]):
                        raise ValueError(f'Error: `{n}` entry of `submodule_params` should either be a different sequence for each group/parallel conv (sequence of sequence of size `len(input_shape) x [dims={self.spatial_dims} or 1]`) or a single sequence of size `1` or `dims={self.spatial_dims}` (then the value is common to all group/parallel convs).{NL}'
                                         f'This is a needed constraint to avoid ambiguous spec in `parallel_convolution`; '
                                         f'E.g., in case you need a `3x2` kernel size for two parallel/group convs you can specify `kernel_size = [3, 2]` but in order to have two parallel/group convs with different kernel sizes of 3x3 and 2x2, then you need to specify: `kernel_size = [[3, 3], [2, 2]]` or alternatively: `[[3], [2]]`.{NL}'
                                         f'I.e. Unlike for regular usage of `torch.nn.Conv*d`, sequence of int(s) are allways needed for "{conv_args_with_sequence_constraint}" conv(s) args; Spec would be ambiguous otherwise')
                    ith_conv_kwargs[n] = v if len(v) > 1 else v[0]
                else:
                    ith_conv_kwargs[n] = v[i] if isinstance(v, Sequence) and len(v) == len(input_shape) else v
            all_conv_ops_kwargs.append(ith_conv_kwargs)

        self.group_convolutions = list()
        for i, (shape, ith_conv_kwargs, act, preact) in enumerate(zip(input_shape, all_conv_ops_kwargs, act_fn, preactivation)):
            conv_op = conv_nd(**ith_conv_kwargs)
            layer_op = layer(layer_op=conv_op, act_fn=act, dropout_prob=dropout_prob, preactivation=preact,
                             norm_type=norm_type, norm_kwargs=norm_kwargs, input_shape=shape[channel_dim:], supported_norm_ops=supported_norm_ops)
            self.group_convolutions.append((f'parallel_conv_{self.spatial_dims}D_{i}', layer_op))

    def forward(self, inputs: TENSOR_OR_SEQ_OF_TENSORS_T) -> TENSOR_OR_SEQ_OF_TENSORS_T:
        # Check if input tensors are valid
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs, ]
        if len(inputs) > len(self.group_convolutions) or (self.raise_on_not_enough_in_tensors and len(inputs) < len(self.group_convolutions)):
            raise ValueError(f'Error: Cant apply `{type(self).__name__}` module with {len(self.group_convolutions)} parallel/group convolutions on {len(inputs)} input tensors.')

        # Apply parallel/group convolutions
        output = list([parallel_conv(in_tensor) for in_tensor, parallel_conv in zip(inputs, self.group_convolutions)])
        return output if len(output) > 1 else output


class MultiresolutionFusion(torch.nn.Module):
    """ Multi-resolution Fusion module as described in [HRNet paper](https://arxiv.org/abs/1908.07919) NN architecture.
    This fusion module can be compared to a regular convolution layer but applied on feature maps with varying resolutions. However, in order to be applied in a 'fully connected' conv. way, each feature maps will be up/down-scaled to each target resolutions (either using bilinear upsampling followed by a 1x1 conv or by applying a 3x3 conv with stride of 2).
    Multi-resolution Fusion modules thus 'fuses' information across all siamese branches (i.e. across all resolutions) of HRNet architecture.

    .. See also related `deepcv.meta.nn.parallel_convolution` module which is also part of basic HRNet architecture building blocks.
    .. See also [`HigherHRNet` paper](https://arxiv.org/pdf/1908.10357.pdf).
    # TODO: Asuchronously yield each output for each branch?
    """

    def __init__(self, input_shape: Union[torch.Size, Sequence[torch.Size]], upscale_conv_kwargs: Dict[str, Any] = None, downscale_conv_kwargs: Dict[str, Any] = None, create_new_branch: bool = True, new_branch_channels: int = None, channel_dim: int = 1, reuse_scaling_convs: bool = True, upscale_align_corners: bool = False):
        """ Instanciate `MultiresolutionFusion` module which 'fuses' information across all multi-resoution siamese branches/streams, like described in [HRNet architecture paper](https://arxiv.org/abs/1908.07919)
        Args:
            - input_shape: Input tensors shapes for each parallel/siamese input branches/streams, with minibatch dim (change `channel_dim` accordingly if needed). All `input_shape` tensor shapes should have the same number of dimensions after `channel_dim` dimension.
            - upscale_conv_kwargs:
            - downscale_conv_kwargs:
            - create_new_branch: Boolean indicating whether to create a new stage by outputing (new parallel branch/stream) by outputing an additional tensor with lower resolution feature maps (See HRNet paper for more details on multiresolution fusion module stages)
            - new_branch_channels: If `create_new_branch`, you can specify how many channels/feature-maps are outputed for this new parallel branch/stream. If `None` (Default), then output the same sumber of channels as previous/upper branch (same kernel/filter count).
            - channel_dim: Channel dim index in provided `input_shape` tensor shapes. Dimensions after channel dim should be spatial features dims on which convolutions are applied (e.g. `input_shape` should have two dims after `channel_dim` for 2D convoltions)
            - reuse_scaling_convs: If `self.reuse_scaling_convs` is `True`, then all up/down scaling convolutions are reused when possible (i.e. weight sharing when (in_channels, out_channels) tuple is the same)

        NOTE: Avoid providing custom `in_channels`, `out_channels` nor `padding` through `upscale_conv_kwargs` and/or `downscale_conv_kwargs` as, by default, those are proccessed automatically from input_shapes and kernel sizes (may be invalid otherwise as `out_channels` must be similar to `in_channel` for a given stream/branch and padding better should keep feature maps size unchanged).
        NOTE: When `self.reuse_scaling_convs` is `True`, having similar number of filters/channels/feature-maps across parallel branches will allow much more weight/parameter sharing than if they are different between parallel streams/branches.
        NOTE: If not excpilitly provided in `downscale_conv_kwargs` and/or `upscale_conv_kwargs`, zero padding is used and padding values are proceessed from kernel sizes in order to obtain unchanged output feature map sizes (See `deepcv.meta.nn.get_padding_from_kernel`).
        """
        if isinstance(input_shape, torch.Size):
            input_shape = [input_shape, ]
        elif len(input_shape) == 0:
            raise ValueError('Error in `deepcv.meta.nn.MultiresolutionFusion`, `input_shape`(s) argument must be a sequence of `torch.Size` with at least one input tensor shape(s). '
                             f'NOTE: If you are performing a single conv on a single tensor shape/resolution, use `deepcv.meta.nn.layer` instead. Got `input_shape="{input_shape}"`')

        # `create_new_branch` indicates whether to create a new conv parallel branch/stream at a lower resoltion (See HRNet paper for more details on multiresolution fusion module)
        self.create_new_branch = create_new_branch
        self.input_shape = input_shape
        self.channel_dim = channel_dim
        self.reuse_scaling_convs = reuse_scaling_convs
        self.spatial_dims = len(input_shape[0][self.channel_dim+1:])  # Assume all input features have the same spatial dims count
        self.align_corners = upscale_align_corners

        upscale_conv_kwargs = self._fill_conv_params(upscale_conv_kwargs, kernel_size=1, dims=self.spatial_dims)
        downscale_conv_kwargs = self._fill_conv_params(downscale_conv_kwargs, kernel_size=3, stride=2, padding_mode='zero', dims=self.spatial_dims)

        # New branch either outputs `new_branch_channels` (if provided), latest/upper branch `out_channels` (if provided) or latest/upper branch `in_channels` features-maps/chanels
        self.new_branch_channels = downscale_conv_kwargs.get('out_channels', input_shape[-1][channel_dim]) if new_branch_channels is None else new_branch_channels
        in_channels = [in_shape[channel_dim] for in_shape in input_shape]
        out_channels = [*in_channels, self.new_branch_channels] if self.create_new_branch else in_channels

        # Define all needed convolutions modules for up/down-scaling between input/output parallel streams/branches
        if self.reuse_scaling_convs:
            # If `self.reuse_scaling_convs` is True, then all up/down scaling convolutions are reused when possible (i.e. weight sharing when (in_channels, out_channels) tuple is the same)
            self.downscaling_3x3_convs = {(in_ch, out_ch): conv_nd(out_channels=out_ch, in_channels=in_ch, **downscale_conv_kwargs)
                                          for in_ch in in_channels for out_ch in out_channels}
            self.upscaling_1x1_convs = {(in_ch, out_ch): conv_nd(out_channels=out_ch, in_channels=in_ch, **upscale_conv_kwargs)
                                        for in_ch in in_channels for out_ch in out_channels}
        else:
            self.upscaling_1x1_convs = [[conv_nd(out_channels=out_ch, in_channels=in_ch, **upscale_conv_kwargs) for in_ch in in_channels] for out_ch in out_channels]
            self.first_downscaling_3x3_convs = [[conv_nd(out_channels=out_ch, in_channels=in_ch, **downscale_conv_kwargs) for in_ch in in_channels] for out_ch in out_channels]
            # We still 'reuse' some convolutions: Additional downsampling convs are used when downscaling by a factor of `stride^2` (4 by default) or more: One more 3x3 conv for each target branches, with `in_channels==out_channels`, is used in addition to the first convs in `first_downscaling_3x3_convs` which are different for each input branches due to different in-channels count (`in_channels` may be different to `out_channels` for the first downscaling convs applied)
            self.additional_downscaling_3x3_convs = [conv_nd(out_channels=out_ch, in_channels=out_ch, **downscale_conv_kwargs) for out_ch in out_channels]

    def _upsample(self, x: torch.Tensor, target_shape: torch.Size, in_branch_idx: int, out_branch_idx: int) -> torch.Tensor:
        """ Upscaling is performed by a bilinear upsampling followed by a 1x1 convolution to match target channel count """
        # Upscale input tensor `x` using (bi/tri)linear interpolation (or 'nearest' interpolation for tensor with 4D or more features maps)
        scale_mode = 'linear' if self.spatial_dims == 1 else ('bilinear' if self.spatial_dims == 2 else ('trilinear' if self.spatial_dims == 3 else 'nearest'))
        x = torch.nn.functional.interpolate(x, size=target_shape[self.channel_dim+1:], mode=scale_mode, align_corners=self.align_corners)

        # Apply 1x1 convolution in order to obtain the target channel/feature-maps count
        if self.reuse_scaling_convs:
            x = self.upscaling_1x1_convs[(x.shape[self.channel_dim], target_shape[self.channel_dim])](x)
        else:
            x = self.upscaling_1x1_convs[out_branch_idx][in_branch_idx](x)
        return x

    def _downsample(self, x: torch.Tensor, out_channels: int, apply_n_times: int, in_branch_idx: int, out_branch_idx: int) -> torch.Tensor:
        """ Downsampling is performed by applying N times a 3x3 convolution with a stride of 2 (by default)
        Each parallel branch/streams process feature maps at a resolution divided by 2, i.e. 4 times less features per map on 2D maps.
        """
        if apply_n_times <= 0:
            raise ValueError(f'Error: received bad argument value, `apply_n_times={apply_n_times}`, in `MultiresolutionFusion._downsample`')

        if self.reuse_scaling_convs:
            for i in range(apply_n_times):
                x = self.downscaling_3x3_convs[(x.shape[1], out_channels)](x)
        else:
            x = self.first_downscaling_3x3_convs[out_branch_idx][in_branch_idx](x)
            for _i in range(apply_n_times - 1):
                x = self.additional_downscaling_3x3_convs[out_branch_idx](x)
        return x

    @classmethod
    def _fill_conv_params(cls, provided_kwargs: Mapping[str, Any], **defaults):
        # Applies default convolutions ops parameters and process padding from kernel size if needed
        conv_kwargs = defaults.update(provided_kwargs)
        if 'padding' not in conv_kwargs:
            conv_kwargs['padding'] = get_padding_from_kernel(conv_kwargs['kernel_size'], warn_on_uneven_kernel=True)
        return conv_kwargs

    def forward(self, inputs: TENSOR_OR_SEQ_OF_TENSORS_T) -> List[torch.Tensor]:
        """ Fusion is performed by summing all down/up-scaled input features from each streams/branches.
        Thus, all inputs are down/up-scaled to all other resolutions before being sumed (plus down-scaled to new branch/stream resolution if `create_new_branch`).
        """
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs, ]
        if len(inputs) != len(self.input_shape):
            raise ValueError(f'Error: Cant apply `{type(self).__name__}` module with {len(self.input_shape)} input parallel branches/streams on {len(inputs)} input tensors.')

        in_resolutions = [np.prod(x.shape[self.channel_dim+1:]) for x in inputs]
        output_shapes = [x.shape for x in ([*inputs, None] if self.create_new_branch else inputs)]

        def _get_downscale_n_times(input_branch_idx: int, out_branch_idx: int) -> int:
            """ Used to get how many times we need to downscale features from `input_branch_idx`th branch to `out_branch_idx`th by using sorted indices of input resolutions (inputs may not be sorted by resolution) """
            return np.where(_get_downscale_n_times.sorted_idx == out_branch_idx) - np.where(_get_downscale_n_times.sorted_idx == input_branch_idx)
        _get_downscale_n_times.sorted_idx = np.argsort(in_resolutions)
        if self.create_new_branch:
            # If we need to create a new branch, then we append its index at the end of sorted indices in order to obtain an additional downscaling
            _get_downscale_n_times.sorted_idx.append(len(in_resolutions))

        outputs = list()
        for out_branch_idx, target_shape in enumerate(output_shapes):
            scaled_features = []
            target_res = None if target_shape is None else np.prod(target_shape[self.channel_dim+1:])
            for in_branch_idx, (other_inputs, in_res) in enumerate(zip(inputs, in_resolutions)):
                if target_shape is None or in_res > target_res:
                    downscale_n_times = _get_downscale_n_times(in_branch_idx, out_branch_idx)
                    out_channels = self.new_branch_channels if out_branch_idx == len(inputs) else target_shape[self.channel_dim]
                    scaled_features.append(self._downsample(other_inputs, apply_n_times=downscale_n_times,
                                                            out_channels=out_channels, out_branch_idx=out_branch_idx, in_branch_idx=in_branch_idx))
                elif in_res < target_res:
                    scaled_features.append(self._upsample(other_inputs, target_shape=target_shape, out_branch_idx=out_branch_idx, in_branch_idx=in_branch_idx))
                else:
                    scaled_features.append(other_inputs)

            outputs.append(torch.sum(scaled_features))
        return outputs


class HRNetV1RepresentationHead(torch.nn.Module):
    def __init__(self, input_shape: Union[torch.Size, Sequence[torch.Size]], channel_dim: int = 1) -> torch.nn.Module:
        if isinstance(input_shape, torch.Size):
            input_shape = [input_shape, ]
        self.input_shape = input_shape
        self.channel_dim = channel_dim

    def _forward(self, inputs: TENSOR_OR_SEQ_OF_TENSORS_T) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs, ]
        if len(inputs) != len(self.input_shape):
            raise ValueError(f'Error: Cant apply `{type(self).__name__}` module with {len(self.input_shape)} input parallel branches/streams on {len(inputs)} input tensors.')
        max_idx = np.argmax(np.prod(x.shape[self.channel_dim+1:] for x in inputs))
        return inputs[max_idx]


class HRNetV2RepresentationHead(torch.nn.Module):
    """ HRNetV2 output representation head. Applies upscaling on lower branch/stream inputs and concatenates those with input from upper/max-resolution branch inputs. """

    def __init__(self, input_shape: Union[torch.Size, Sequence[torch.Size]], repr_mix_conv_kwargs: Dict[str, Any] = None, channel_dim: int = 1, upscale_align_corners: bool = False) -> torch.nn.Module:
        """ 
        NOTE: `out_channels` must be specified in `repr_mix_conv_kwargs` (1x1 conv applied on concatenated (upscaled) representations to obtain target channel count and mix informations from all representations)
        """
        if isinstance(input_shape, torch.Size):
            input_shape = [input_shape, ]
        self.input_shape = input_shape
        self.channel_dim = channel_dim
        max_shape_idx = np.argmax(np.prod(shape[self.channel_dim+1:] for shape in input_shape))
        self.outout_spatial_shape = input_shape[max_shape_idx][channel_dim+1:]
        self.outout_channels = repr_mix_conv_kwargs['out_channels']
        self.spatial_dims = len(input_shape[0][channel_dim+1:])
        self.scale_mode = 'linear' if self.spatial_dims == 1 else ('bilinear' if self.spatial_dims == 2 else ('trilinear' if self.spatial_dims == 3 else 'nearest'))
        self.align_corners = upscale_align_corners

        # Define upscaling convolutions
        in_channels = sum([shape[channel_dim] for shape in input_shape])
        repr_mix_conv_kwargs = MultiresolutionFusion._fill_conv_params(repr_mix_conv_kwargs, kernel_size=1, dims=self.spatial_dims)
        self.repr_mix_1x1_conv = conv_nd(in_channels=in_channels, **repr_mix_conv_kwargs)

    def forward(self, inputs: TENSOR_OR_SEQ_OF_TENSORS_T) -> TENSOR_OR_SEQ_OF_TENSORS_T:
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs, ]
        if len(inputs) != len(self.input_shape):
            raise ValueError(f'Error: Cant apply `{type(self).__name__}` module with {len(self.input_shape)} input parallel branches/streams on {len(inputs)} input tensors.')

        # Upscale (interpolation only) input features from lower branches
        outputs = list()
        for branch_in in inputs:
            if np.prod(branch_in.shape[self.channel_dim+1:]) < np.prod(self.outout_spatial_shape):
                branch_in = torch.nn.functional.interpolate(branch_in, size=self.outout_spatial_shape, mode=self.scale_mode, align_corners=self.align_corners)
            outputs.append(branch_in)

        # Concatenate each upscaled representations with upper/max-resolution branch/stream input tensor and apply a 1x1 conv to mix them and obtain target channel count
        return self.repr_mix_1x1_conv(torch.cat(outputs, dim=self.channel_dim))


class HRNetV2pRepresentationHead(HRNetV2RepresentationHead):
    def __init__(self, input_shape: Union[torch.Size, Sequence[torch.Size]], repr_mix_conv_kwargs: Dict[str, Any] = None, downscale_conv_kwargs: Dict[str, Any] = None, max_downscaling_count: int = None, channel_dim: int = 1) -> torch.nn.Module:
        """
        NOTE: Avoid providing `out_channels` nor `in_channels` in `downscale_conv_kwargs`. Those are already set to `repr_mix_conv_kwargs['out_channels']`
        """
        super().__init__(input_shape=input_shape, repr_mix_conv_kwargs=repr_mix_conv_kwargs, channel_dim=channel_dim)
        self.downscaling_count = len(input_shape) - 1 if max_downscaling_count is None else max_downscaling_count
        downscale_conv_kwargs = self._fill_conv_params(downscale_conv_kwargs, kernel_size=3, stride=2, padding_mode='zero', dims=self.spatial_dims)
        self.downscale_conv = conv_nd(out_channels=self.outout_channels, in_channels=self.outout_channels, **downscale_conv_kwargs)

    def forward(self, inputs: TENSOR_OR_SEQ_OF_TENSORS_T) -> TENSOR_OR_SEQ_OF_TENSORS_T:
        hrnetv2_out = super().forward(inputs)
        input_shapes = np.sort([x.shape[self.channel_dim:] for x in inputs])
        del inputs  # free memory from inputs as they are no longer needed

        # Downscale concatenated representations from HRNetV2 head to all branch resolutions/spatial-shapes with 3x3 2-strided conv (by default)
        outputs = [hrnetv2_out, ]
        for _i in range(self.downscaling_count):
            outputs.append(self.downscale_conv(outputs[-1]))
        return outputs


# TODO: HRNetInputStem
def hrnet_input_stem(input_shape: torch.Size, submodule_params: Dict[str, Tuple[Sequence[Any], Any]], conv_count: int = 2, act_fn: Union[Sequence[Optional[Type]], Optional[Type]] = torch.nn.ReLU,
                     preactivation: Union[Sequence[bool], bool] = False, dropout_prob: float = None, norm_type: Sequence[NormTechnique] = None, norm_kwargs: Sequence[Dict[str, Any]] = None,
                     supported_norm_ops: NORM_TECHNIQUES_MODULES_T = None, channel_dim: int = 1) -> torch.nn.Module:
    """ Input stem block as described in [HRNet NN architecture](https://arxiv.org/abs/1908.07919).
    This input module consists of `conv_count` (2 by defaults) 3x3 2-stride convolutions. Hence, input stem decreases input resolution by a `1/2^N` factor (`1/4` by default).
    NOTE: `submodule_params` have default values for all convolution ops arguments, except for `out_channels` which is needed (filter count). Avoid providing `in_channels` as this is already known from `input_shape`.
    """
    assert conv_count > 0, f'Error in `hrnet_input_stem`, `conv_count` argument must be greater than 0, got `conv_count={conv_count}`'
    conv_defaults = dict(kernel_size=(3, 3), stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
    conv_defaults.update(submodule_params)
    if 'in_channels' not in submodule_params:
        conv_defaults['in_channels'] = input_shape[channel_dim]

    strided_convs = list()
    for i in range(conv_count):
        conv_op = torch.nn.Conv2d(**conv_defaults)
        layer_op = layer(layer_op=conv_op, act_fn=act_fn, dropout_prob=dropout_prob, preactivation=preactivation,
                         norm_type=norm_type, norm_kwargs=norm_kwargs, input_shape=input_shape[channel_dim:], supported_norm_ops=supported_norm_ops)
        strided_convs.append((f'stem{"_strided" if conv_defaults["stride"] > 1 else ""}_conv_{i}', layer_op))
    return torch.nn.Sequential(OrderedDict(strided_convs))


#____________________________________________________ UNIT TESTS ______________________________________________________#

if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
