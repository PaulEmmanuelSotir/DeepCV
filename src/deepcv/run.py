#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Application entry point."""
import os
import re
import logging
import functools
from pathlib import Path
from typing import Dict, Union, Any, Type

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.framework.hooks import hook_impl
import kedro.io.transformers as transformers
from kedro.framework.context import KedroContext, load_package_context

import git
import torch
import ignite
import mlflow
import anyconfig
import torchvision

import deepcv

__author__ = 'Paul-Emmanuel Sotir'


class ProjectMainHooks:
    """ Project main hooks
    NOTE: For MLflow experiments/runs tracking support, pipeline(s) (or at least one node of the pipeline(s)) which involves training should have a 'train' tag (project hooks defined in `deepcv.run` creates/ends mlflow run for each `train` pipelines)
    """

    def __init__(self, project_ctx: KedroContext):
        self.project_ctx = project_ctx

    def _mlflow_run_repo_info(self):
        """ This code creates special mlflow tags about DeepCV repository info, which is not done by mlflow when starting MLFlow run from code instead of `mlflow run` command
        Code similar to (mlflow.projects._create_run)[https://www.mlflow.org/docs/latest/_modules/mlflow/projects.html] which is not called by `mlflow.start_run`
        """
        tags = {mlflow.utils.mlflow_tags.MLFLOW_SOURCE_NAME: self.project_ctx.project_name,
                mlflow.utils.mlflow_tags.MLFLOW_SOURCE_TYPE: mlflow.entities.SourceType.to_string(mlflow.entities.SourceType.PROJECT),
                mlflow.utils.mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT: 'src/deepcv/run.py'}
        try:
            repo = git.Repo(self.project_ctx.project_path, search_parent_directories=True)
            git_repo_url = repo.remote().url if 'origin' in repo.remotes else (repo.remotes[0].url if len(repo.remotes) > 0 else '')
            git_repo_url = re.sub(r'git@([.\w]+):', r'https://\1/', git_repo_url).rstrip('.git')  # Convert SSH git URL to http URL
            mlflow.log_param('commit_url', git_repo_url + f'/commit/{repo.head.commit.hexsha}/')

            # We also update MLFLOW_SOURCE_NAME to repo URL so that MLFlow web UI is able to parse it and render commit and source hyperlinks (MLFLow only supports github URLs for now)
            tags.update({mlflow.utils.mlflow_tags.MLFLOW_SOURCE_NAME: git_repo_url,
                         mlflow.utils.mlflow_tags.MLFLOW_GIT_BRANCH: repo.active_branch.name,
                         mlflow.utils.mlflow_tags.MLFLOW_GIT_REPO_URL: git_repo_url,
                         mlflow.utils.mlflow_tags.MLFLOW_GIT_COMMIT: repo.head.commit.hexsha})
        except (ImportError, git.InvalidGitRepositoryError, git.GitCommandNotFound, ValueError, git.NoSuchPathError) as e:
            logging.warning(f'Failed to import Git or to get repository informations. Error: {e}')

        mlflow.set_tags(tags)

    @ hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog):
        node_tags = functools.reduce(set.union, [n.tags for n in pipeline.nodes])
        if 'train' in run_params['tags'] or 'train' in node_tags:
            if mlflow.active_run() is None:
                # Create MLFlow run in an experiment named after pipeline involved in training and log various pipeline/datasets informations to mlflow
                # TODO: find another way to name experiment as pipeline name is only available when running `kedro run --pipeline=<pipeline_name>` (e.g. special tag to node after which experiment is named)
                pipeline_name = run_params['pipeline_name'] if run_params['pipeline_name'] else 'unknown_pipeline'
                mlflow.set_experiment(f'{self.project_ctx.project_name}_{pipeline_name}')
                mlflow.start_run(run_name=f'{pipeline_name}_run_{run_params["run_id"]}')
                self._mlflow_run_repo_info()
                mlflow.set_tags({f'kedro_node_tag_{i}': tag for i, tag in enumerate(node_tags)})
                mlflow.log_params({n: v for n, v in run_params.items() if v})
                mlflow.log_param('pipeline.json', pipeline.to_json())
                mlflow.log_param('pipeline.describe', pipeline.describe())
                mlflow.log_param('pipeline.pipeline_datasets', pipeline.data_sets())

    @ hook_impl
    def after_pipeline_run(self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog):
        if 'train' in run_params['tags'] or any(['train' in n.tags for n in pipeline.nodes]):
            if mlflow.active_run() is not None:
                mlflow.end_run()

    @ hook_impl
    def on_pipeline_error(self, error: Exception, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog):
        if 'train' in run_params['tags'] or any(['train' in n.tags for n in pipeline.nodes]):
            if mlflow.active_run() is not None:
                mlflow.end_run()

    @ hook_impl
    def before_node_run(self, node: Node, catalog: DataCatalog, inputs: Dict[str, Any], is_async: bool, run_id: str):
        pass

    @ hook_impl
    def after_node_run(self, node: Node, catalog: DataCatalog, inputs: Dict[str, Any], outputs: Dict[str, Any], is_async: bool, run_id: str):
        pass

    @ hook_impl
    def on_node_error(self, error: Exception, node: Node, catalog: DataCatalog, inputs: Dict[str, Any], is_async: bool, run_id: str):
        pass

    @ hook_impl
    def after_catalog_created(self, catalog: DataCatalog, conf_catalog: Dict[str, Any], conf_creds: Dict[str, Any], feed_dict: Dict[str, Any], save_version: str, load_versions: Dict[str, str], run_id: str):
        pass


class ProjectContext(KedroContext):
    """ Users can override the remaining methods from the parent class here, or create new ones (e.g. as required by plugins) """

    project_name = "DeepCV"
    project_version = "0.16.1"  # `project_version` is the version of kedro used to generate the project
    package_name = "deepcv"

    hooks = tuple()

    def __init__(self, project_path: Union[Path, str], env: str = None, extra_params: Dict[str, Any] = None):
        self.hooks += (ProjectMainHooks(self),)
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
        # TODO: find profile time profiler replacement in Kedro 0.16.1 (was available in contrib subpackage of kedro <0.16.0)
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
