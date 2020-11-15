#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Construction of the master pipeline.
"""
import logging
import operator
from pathlib import Path
from functools import reduce
from typing import Dict, Union, Any, Optional, Callable

from kedro.pipeline import Pipeline, node
import kedro.pipeline.decorators as dec

import deepcv.utils
import deepcv.meta
import deepcv.keypoints
import deepcv.detection
import deepcv.classification

__all__ = ['create_pipelines', 'DECORATORS', 'GET_PIPELINE_FN_NAME', 'SUBPACKAGES_WITH_PIPELINES']
__author__ = 'Paul-Emmanuel Sotir'


DECORATORS = [dec.log_time]  # Other decorator available: memory_profiler? ,retry, spark_to_pandas, pandas_to_spark
GET_PIPELINE_FN_NAME = 'get_pipelines'
SUBPACKAGES_WITH_PIPELINES = [deepcv.classification, deepcv.keypoints, deepcv.detection]


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.
    Args:
        kwargs: Ignore any additional arguments added in the future.
    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    NOTE: For MLflow experiments/runs tracking support, pipeline(s) (or at least one node of the pipeline(s)) which involves training should have a 'train' tag (project hooks defined in `deepcv.run` creates/ends mlflow run for each `train` pipelines)
    """
    pipeline_map = {}

    for subpackage in SUBPACKAGES_WITH_PIPELINES:
        get_pipelines: Optional[Callable[[None], Dict[str, Pipeline]]] = getattr(subpackage, GET_PIPELINE_FN_NAME, None)
        if get_pipelines is None:
            logging.warn(f'Warning: Could\'t find `{GET_PIPELINE_FN_NAME}` function in `{subpackage}` Deepcv subpackage or submodule.')
        pipeline_map.update({n: p.decorate(*DECORATORS) for n, p in get_pipelines().items()})

    return {**pipeline_map, "__default__": reduce(operator.add, pipeline_map.values())}


if __name__ == '__main__':
    # Simply call `deepcv.pipelines.create_pipelines`
    pipelines = create_pipelines()
