#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Construction of the master pipeline.
"""
from typing import Dict

from kedro.pipeline import Pipeline
import kedro.pipeline.decorators as dec

from deepcv.pipelines import machine_learning as ml
from deepcv.pipelines import data_engineering as de

DECORATORS = [dec.log_time, dec.mem_profile]  # Other decorator available: retry, spark_to_pandas, pandas_to_spark

__author__ = 'Paul-Emmanuel Sotir'


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.
    Args:
        kwargs: Ignore any additional arguments added in the future.
    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    data_engineering_pipeline = de.create_pipeline()
    machine_learning_pipeline = ml.create_pipeline()

    return {"de": data_engineering_pipeline,
            "ml": machine_learning_pipeline,
            "__default__": (data_engineering_pipeline + machine_learning_pipeline).decorate(*DECORATORS)}
