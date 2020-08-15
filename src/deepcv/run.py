#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Application entry point."""
import os
import logging
import functools
from pathlib import Path
from typing import Dict, Union, Any

from kedro.pipeline import Pipeline
import kedro.io.transformers as transformers
from kedro.framework.context import KedroContext, load_package_context

import torch
import ignite
import mlflow
import anyconfig
import torchvision

import deepcv

__author__ = 'Paul-Emmanuel Sotir'


class ProjectContext(KedroContext):
    """ Users can override the remaining methods from the parent class here, or create new ones (e.g. as required by plugins) """

    project_name = "DeepCV"
    project_version = "0.16.4"  # `project_version` is the version of kedro used to generate the project
    package_name = "deepcv"

    hooks = tuple()

    def __init__(self, project_path: Union[Path, str], env: str = None, extra_params: Dict[str, Any] = None):
        self.hooks += (deepcv.hooks.ProjectMainHooks(self),)
        super().__init__(project_path=project_path, env=env, extra_params=extra_params)

        # We use Ruamel.yaml backend and monkey-patch it to parse YAML configurations files and allow Python types to be constructed from YAML (anyconfig is used by Kedro to load YAML config files)
        deepcv.utils.set_anyconfig_yaml_parser_priorities(pyyaml_priority=30, ryaml_priority=100)
        anyconfig.load = functools.partial(anyconfig.load, typ='unsafe')

        # Setup MLFlow tracking
        self.mlflow_tracking_uri = (self._project_path / r'data/04_training/mlruns').relative_to(Path.cwd())
        self.mlflow_tracking_uri.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(str(self.mlflow_tracking_uri))

    def _get_pipelines(self) -> Dict[str, Pipeline]:
        return deepcv.pipeline.create_pipelines()

    def _create_catalog(self, *args, **kwargs):
        catalog = super()._create_catalog(*args, **kwargs)
        # TODO: find profile time profiler replacement in Kedro 0.16.4 (was available in contrib subpackage of kedro <0.16.0)
        # profile_time = transformers.ProfileTimeTransformer()  # instantiate a built-in transformer
        # catalog.add_transformer(profile_time)  # apply it to the catalog
        return catalog


def run_package():
    # Entry point for running a Kedro project packaged with `kedro package` using `python -m <project_package>.run` command.
    project_context = load_package_context(project_path=Path.cwd(), package_name=Path(__file__).resolve().parent.name)
    project_context.run()


if __name__ == "__main__":
    # Entry point for running pip-installed projects using `python -m <project_package>.run` command
    run_package()
