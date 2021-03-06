#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Kedro project hooks/callbacks: allows to define custom behavior on creation/ending/errors of pipelines/nodes/data-catalog.
For example, MLFlow support is improved by creating/ending a MLFlow run/experiment when executing a pipeline tagged with 'train'.
"""
import re
import inspect
import logging
import functools
from typing import Dict, Union, Any

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.framework.hooks import hook_impl
from kedro.framework.context import KedroContext

import git
import nni
import mlflow
import configparser

import deepcv.meta

__all__ = ['ProjectMainHooks']
__author__ = 'Paul-Emmanuel Sotir'


class ProjectMainHooks:
    """ Project main hooks
    NOTE: For MLflow experiments/runs tracking support, pipeline(s) (or at least one node of the pipeline(s)) which involves training should have a 'train' tag (project hooks defined in `deepcv.run` creates/ends mlflow run for each `train` pipelines)
    """

    def __init__(self, project_ctx: KedroContext):
        self.project_ctx = project_ctx

    @hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog):
        self._start_mlflow_run(run_params, pipeline)

    @hook_impl
    def after_pipeline_run(self, run_params: Dict[str, Any], run_result: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog):
        if 'train' in run_params['tags'] or any(['train' in n.tags for n in pipeline.nodes]):
            if mlflow.active_run() is not None:
                mlflow.end_run()

    @hook_impl
    def on_pipeline_error(self, error: Exception, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog):
        if 'train' in run_params['tags'] or any(['train' in n.tags for n in pipeline.nodes]):
            if mlflow.active_run() is not None:
                mlflow.end_run()

    @hook_impl
    def before_node_run(self, node: Node, catalog: DataCatalog, inputs: Dict[str, Any], is_async: bool, run_id: str):
        pass

    @hook_impl
    def after_node_run(self, node: Node, catalog: DataCatalog, inputs: Dict[str, Any], outputs: Dict[str, Any], is_async: bool, run_id: str):
        pass

    @hook_impl
    def on_node_error(self, error: Exception, node: Node, catalog: DataCatalog, inputs: Dict[str, Any], is_async: bool, run_id: str):
        pass

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog, conf_catalog: Dict[str, Any], conf_creds: Dict[str, Any], feed_dict: Dict[str, Any], save_version: str, load_versions: Dict[str, str], run_id: str):
        pass

    def _start_mlflow_run(self, run_params: Dict[str, Any], pipeline: Pipeline):
        """ Log basic informations to MLFlow about pipeline if this pipeline is tagged with 'train' (creates a new MLFLow experiment and/or run named after training pipeline if it doesn't exists yet)
        NOTE: If NNI is in dry run mode (mode used to generate NNI Classic NAS search space JSON file from a model which contains NNI NAS Mutables `LayerChoice` and/or `InputChoice`) we avoid creating any new MLFlow experiment/run nor logging anything else to mlflow during this dry run
        """
        node_tags = functools.reduce(set.union, [n.tags for n in pipeline.nodes])
        if not deepcv.meta.nni_tools.is_nni_gen_search_space_mode() and ('train' in run_params['tags'] or 'train' in node_tags):
            if mlflow.active_run() is None:
                # Create MLFlow run in an experiment named after pipeline involved in training and log various pipeline/datasets informations to mlflow. If we are running an NNI hp/nas search, mlflow experiment and run will be named after NNI experiment and trial ids for better consitency.
                # TODO: find another way to name experiment as pipeline name is only available when running `kedro run --pipeline=<pipeline_name>` (e.g. special tag to node after which experiment is named)

                if not deepcv.meta.nni_tools.is_nni_run_standalone():  # 'STANDALONE' is NNI default experiment ID if python process haven't been started by NNI
                    nni_experiment = nni.get_experiment_id()
                    mlflow.set_experiment(nni_experiment)
                    mlflow.start_run(run_name=nni.get_trial_id())
                    # Flag indicating whether we are using NNI HP or Classic NAS API (Hyperparameter and/or Classic Neural Architecture search using NNI)
                    mlflow.set_tag('nni_standalone_mode', False)
                    mlflow.set_tag('nni_experiment_id', nni_experiment)
                    mlflow.set_tag('nni_trial_id', nni.get_trial_id())
                    mlflow.set_tag('nni_sequence_id', nni.get_sequence_id())
                else:
                    pipeline_name = run_params['pipeline_name'].lower() if run_params['pipeline_name'] else 'default'
                    mlflow.set_experiment(f'{self.project_ctx.project_name.lower()}_{pipeline_name}')
                    mlflow.start_run(run_name=f'{pipeline_name.lower()}_run_{run_params["run_id"]}')
                    mlflow.set_tag('nni_standalone_mode', True)

            # Log basic informations about Kedro training pipeline to mlflow
            mlflow.set_tags({f'kedro_node_tag_{i}': tag for i, tag in enumerate(node_tags)})
            mlflow.log_params({n: v for n, v in run_params.items() if v})
            mlflow.log_param('pipeline.json', pipeline.to_json())
            mlflow.log_param('pipeline.describe', pipeline.describe())
            mlflow.log_param('pipeline.pipeline_datasets', pipeline.data_sets())

            """ The following code creates special mlflow tags about current repository infos, which is not done by mlflow when starting an MLFlow run from code instead of from `mlflow run` command
            Code inspired from [`mlflow.projects._create_run`](https://www.mlflow.org/docs/latest/_modules/mlflow/projects.html) which doesn't seems to be called by `mlflow.start_run`
            """
            tags = {mlflow.utils.mlflow_tags.MLFLOW_SOURCE_NAME: self.project_ctx.package_name,
                    mlflow.utils.mlflow_tags.MLFLOW_SOURCE_TYPE: mlflow.entities.SourceType.to_string(mlflow.entities.SourceType.PROJECT),
                    mlflow.utils.mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT: inspect.getsourcefile(type(self.project_ctx))}
            try:
                repo = git.Repo(self.project_ctx.project_path, search_parent_directories=True)
                git_repo_url = repo.remote().url if 'origin' in repo.remotes else (repo.remotes[0].url if len(repo.remotes) > 0 else '')
                git_repo_url = re.sub(r'git@([.\w]+):', r'https://\1/', git_repo_url).rstrip('.git')  # Convert SSH git URL to http URL
                mlflow.log_param('commit_url', git_repo_url + f'/commit/{repo.head.commit.hexsha}/')

                # We also set MLFLOW_SOURCE_NAME to repo URL so that MLFlow web UI is able to parse it and render commit and source hyperlinks (MLFLow only supports github URLs for now)
                tags.update({mlflow.utils.mlflow_tags.MLFLOW_SOURCE_NAME: git_repo_url if git_repo_url else self.project_ctx.project_name,
                             mlflow.utils.mlflow_tags.MLFLOW_GIT_BRANCH: repo.active_branch.name,
                             mlflow.utils.mlflow_tags.MLFLOW_GIT_REPO_URL: git_repo_url,
                             mlflow.utils.mlflow_tags.MLFLOW_GIT_COMMIT: repo.head.commit.hexsha})

                # Change mlflow user to be git repository user instead of system user (if any git user is specified)
                git_config_reader = repo.config_reader()
                git_config_reader.read()
                user = git_config_reader.get_value('user', 'name', default=None)
                email = git_config_reader.get_value('user', 'email', default=None)
                if user or email:
                    tags[mlflow.utils.mlflow_tags.MLFLOW_USER] = (str(user) + (f' <{email}>' if email else '')) if user else str(email)
            except (ImportError, OSError, ValueError, IOError, KeyError, git.GitError, configparser.Error) as e:
                logging.warning(f'Failed to import Git or to get repository informations. Error: {e}')

            mlflow.set_tags(tags)
