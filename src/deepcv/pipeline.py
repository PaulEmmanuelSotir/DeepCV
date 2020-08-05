#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Construction of the master pipeline.
"""
import operator
from pathlib import Path
from functools import reduce
from typing import Dict, Union, Any

from kedro.pipeline import Pipeline, node
import kedro.pipeline.decorators as dec

import deepcv.classification.image
import deepcv.keypoints.detector
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
    pipeline_mapping.update({n: p.decorate(*DECORATORS) for n, p in deepcv.classification.image.get_img_classifier_pipelines().items()})
    return {**pipeline_mapping, "__default__": reduce(operator.add, [p for n, p in pipeline_mapping.items()])}
