#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Construction of the master pipeline.
"""
import operator
from typing import Dict
from functools import reduce

from kedro.pipeline import Pipeline
import kedro.pipeline.decorators as dec

import deepcv.detection.object

DECORATORS = [dec.log_time]  # Other decorator available: memory_profiler? ,retry, spark_to_pandas, pandas_to_spark

__author__ = 'Paul-Emmanuel Sotir'


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.
    Args:
        kwargs: Ignore any additional arguments added in the future.
    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    NOTE: For MLflow experiments/runs tracking support, pipeline(s) (or at least one node of the pipeline(s)) which involves training should have a 'train' tag (project hooks defined in `deepcv.run` creates/ends mlflow run for each `train` pipelines)
    """
    pipeline_mapping = {}
    pipeline_mapping.update({n: p.decorate(*DECORATORS) for n, p in deepcv.detection.object.get_object_detector_pipelines().items()})
    return {**pipeline_mapping, "__default__": reduce(operator.add, [p for n, p in pipeline_mapping.items()])}
