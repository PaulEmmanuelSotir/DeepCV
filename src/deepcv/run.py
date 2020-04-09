#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Application entry point."""
import mlflow
from pathlib import Path
from typing import Dict

from kedro.context import KedroContext, load_context
from kedro.pipeline import Pipeline
import kedro.extras.transformers as transformers

from .pipeline import create_pipelines
from .utils import source_dir

__author__ = 'Paul-Emmanuel Sotir'


class ProjectContext(KedroContext):
    """ Users can override the remaining methods from the parent class here, or create new ones (e.g. as required by plugins)
    """

    project_name = "DeepCV"
    project_version = "0.15.7"

    def _get_pipelines(self) -> Dict[str, Pipeline]:
        return create_pipelines()

    def _create_catalog(self, *args, **kwargs):
        catalog = super()._create_catalog(*args, **kwargs)
        profile_time = transformers.ProfileTimeTransformer()  # instantiate a built-in transformer
        catalog.add_transformer(profile_time)  # apply it to the catalog
        return catalog


def run_package():
    # entry point for running pip-install projects
    # using `<project_package>` command
    mlflow.set_tracking_uri(source_dir(__file__).joinpath(r'../../MLRuns/'))
    project_context = load_context(Path.cwd())
    project_context.run()


if __name__ == "__main__":
    # entry point for running pip-installed projects
    # using `python -m <project_package>.run` command
    run_package()
