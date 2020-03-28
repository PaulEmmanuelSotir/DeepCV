#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example code for the nodes in the example pipeline. This code is meant just for illustrating basic Kedro features.
"""

from kedro.pipeline import Pipeline, node

from .nodes import preprocess_iris

__author__ = 'Paul-Emmanuel Sotir'


def create_pipeline(**kwargs):
    return Pipeline([node(func=preprocess_iris, inputs=['example_iris_data', 'params:example_test_data_ratio'],
                          outputs={'train_x': 'example_train_x', 'train_y': 'example_train_y', 'test_x': 'example_test_x', 'test_y': 'example_test_y'},
                          name='preprocess_iris_dataset')],
                    name='iris_preprocess')
