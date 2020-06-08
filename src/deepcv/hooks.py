#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Kedro project hooks/callbacks: allows to define custom behavior on creation/ending/errors of pipelines/nodes/data-catalog.
For example, MLFlow support is improved by creating/ending a MLFlow run/experiment when executing a pipeline tagged with 'train'. 
"""
import re
import logging
import functools
from typing import Dict, Union, Any

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.framework.hooks import hook_impl
from kedro.framework.context import KedroContext

import git
import mlflow

__all__ = ['ProjectMainHooks']


class ProjectMainHooks:
    """ Project main hooks
    NOTE: For MLflow experiments/runs tracking support, pipeline(s) (or at least one node of the pipeline(s)) which involves training should have a 'train' tag (project hooks defined in `deepcv.run` creates/ends mlflow run for each `train` pipelines)
    """

    def __init__(self, project_ctx: KedroContext):
        self.project_ctx = project_ctx

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
