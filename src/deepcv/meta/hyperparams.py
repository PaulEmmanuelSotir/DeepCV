#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Hyperparameter search meta module - hyperparams.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir

# To-Do List
    - TODO: implement tools for NNI (https://github.com/microsoft/nni) usage (NNI Board and NNICTL) + MLFlow versionning and viz
    - TODO: For hyperparameter embedding: read about graph embedding techniques like: https://github.com/google-research/google-research/tree/master/graph_embedding/ddgk and https://github.com/google-research/google-research/tree/master/graph_embedding/watch_your_step
"""
import uuid
import json
import types
import logging
import functools
import collections
import multiprocessing
from pathlib import Path
from typing import Sequence, Iterable, Callable, Dict, Tuple, Any, Union, Optional, List, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import nni
import mlflow
import networkx
import anyconfig
import numpy as np
from scipy.optimize import least_squares

import deepcv.utils
import deepcv.meta.nn
import deepcv.meta.data.datasets
import deepcv.meta.data.training_metadata


__all__ = ['Hyperparameters', 'HyperparameterSpace', 'HyperparamsEmbedding', 'GeneralizationAcrossScalesPredictor', 'to_hyperparameters',
           'merge_hyperparameters', 'hp_search', 'sample_nni_hp_space', 'get_hp_position_in_search_space', 'generate_hp_space_template']
__author__ = 'Paul-Emmanuel Sotir'

Hyperparameters = deepcv.meta.data.training_metadata.Hyperparameters
HyperparameterSpace = deepcv.meta.data.training_metadata.HyperparameterSpace


class HyperparamsEmbedding(nn.Module):
    """ Hyper-parameter dict embedding module
    Given an hyper-parameter space (hp_space usually used during smpling of hyperopt's hyperparameter search), converts input hyperparameter dict into a vectorized representation.
    Applied to a valid hp dict, returns a fixed-size vector embedding which can be interpreted as vectors in an euclidian space (final neural net layers are enforced by loss constraints to output embeddings in euclidian-like space)
    """

    def __init__(self, embedding_size: int, intermediate_embedding_size: Optional[int] = 128, hp_space: Optional[HyperparameterSpace] = None):
        self._embedding_size = embedding_size
        self._intermediate_embedding_size = max(embedding_size + 32, intermediate_embedding_size)
        self._hp_space = hp_space

        # Define a simple shallow neural net architecture used to obtain final hyper-parameter embedding in appropriate space (euclidian space easier to interpret than space of `_from_hp_space(hp)` vector)
        mean_embedding_size = sum(self._intermediate_embedding_size, self.embedding_size) // 2
        linear1 = nn.Linear(in_features=self._intermediate_embedding_size + 1, out_features=mean_embedding_size)  # + 1 for hp dict hash input
        linear2 = nn.Linear(in_features=mean_embedding_size, out_features=min(mean_embedding_size, self.embedding_size * 2))
        linear3 = nn.Linear(in_features=min(mean_embedding_size, self.embedding_size * 2), out_features=self.embedding_size)
        self._net = nn.Sequential(linear1, nn.ReLU(), linear2, nn.ReLU(), linear3, nn.Softmax())

    def fit(self, hp_dicts):
        # Unsupervised training to learn 3-layers fully connected NN for hp embedding in euclidian-like space
        raise NotImplementedError

    def _from_hp_space(self, hp: Dict[str, Any], _hp_repr: Optional[np.ndarray] = None):
        # TODO: parse hp space and append as much as possible (so that whole repr fits in 'embedding_size') binary representation of each relative positions in hp_space ranges/choices
        # TODO: refactor this code (use NNI's hp_space generated YAML file instead of hyperopt)
        # for node in hyperopt.vectorize.uniq(hyperopt.vectorize.toposort(self._hp_space)):
        #     if isinstance(node, hyperopt.tpe.pyll.rec_eval):
        #         pass
        if not _hp_repr:
            # First call of recursion over hp_space dict
            _hp_repr = np.array([])
        for n, v in hp.items():
            if isinstance(v, Dict):
                _hp_repr.append(self._from_hp_space(v, _hp_repr))
            elif True or isinstance(v, ...):
                raise NotImplementedError
                self._hp_space[n]
                _hp_repr.append(...)
        return torch.from_numpy(_hp_repr)

    def forward(self, hp: Hyperparameters) -> torch.Tensor:
        # Parse hp_space to deduce hyperparameter dict's position in hp_space: returns an array of relative positions (or None value(s) if hp_space defines choice(s)/range(s) which are not present in input hp dict)
        if self._hp_space is not None:
            hp_repr = self._from_hp_space(hp)

        # Concat binary representations of all relative positions of hp_repr into a vector of size 'embdding_size'
        bits_per_position = 32 * self._intermediate_embedding_size // len(hp_repr)  # TODO: replace 32 with sizeof(int)
        intermediate_embedding = np.ndarray((self._intermediate_embedding_size,), dtype=np.float32)
        for i, pos in enumerate(hp_repr):
            # TODO: refactor this
            # highest_bit_idx = np.floor(np.log(pos, 2))
            # bit_pos_repr = pos & (sum(np.power(2, k) for k in range(bits_per_position)) << (
            #     highest_bit_idx - bits_per_position))  # TODO: fix it by making sure that pos is a positive integer

            j = np.mod(i * bits_per_position, 32)
            # TODO: handle bit-level offset and consequences
            intermediate_embedding[j]

        # Concat obtained embedding vector with hp dict hash and input it to FC NN
        # TODO: append an embedding of hp graph topology to intermediate embedding using networkx spectral embedding of hp_space nodes
        hp_repr = torch.cat(hp_repr, torch.Tensor(hash(hp)))
        return self._net(hp_repr)

    def _topologic_hp_embedding(self, hp: Dict[str, Any], topo_embedding_size=32):
        # TODO: refactor this code (use NNI's hp_space generated YAML file instead of hyperopt)
        G = networkx.DiGraph()
        # nodes = hyperopt.vectorize.dfs(expr)

        nodes_dict = networkx.spectral_layout(G, center=(0, 0), dim=2)
        topo_embedding = np.array([], dtype=np.float32)
        for n, v in nodes_dict.items():
            raise NotImplementedError
            # topo_embedding.append()
        return topo_embedding


class GeneralizationAcrossScalesPredictor(nn.Module):
    """ GeneralizationAcrossScalesPredictor
    Improved implementation of [a constructive prediction of the generalization error across scales](https://arxiv.org/pdf/1909.12673.pdf), which can combine proposed paper's model with a two layer fully connected neural net to better predict valid loss (optional).
    By default, validation error is predicted by performing a least squares regression of `GeneralizationAcrossScalesPredictor.error_landscape_estimation` enveloppe function which depends on a few parameters (max. 6 parameters to fit).
    This lightweight model allows to predict model's best valid loss landscape from very few model training example. This can be usefull, for example, to perform a faster hyperparameter search by training a model a few times on small trainset subsets and estimate how hyperparmeters would perform on a full trainset training.
    This model also takes into account the influence of simple model capacity changes over validation error landscape if regression is done on varying (model capacity/dataset size/best validation error) triplets.
    We modified validation error landscape modeling in order to reduce it's parameter count if model capacity or dataset size doesn't change across given training results, which permits to perform less training results to accuralty regress on validation error landscape, depending on which use you make of this model.
    Moreover, optional fully connected neural net can improve these validation error prediction across more diverse setups if given enought training data. This additional model can eventually take various dataset statistics, hyperparameter embedding, training loss curves (not only best valid/train losses) as input.
    But in order to make efficient use of the fully connected model, you will need to fit it on much more training results than basic/lightweight model.
    # TODO: scale neural net model based on how much training data is available
    # TODO: predict best losses with their respective uncertainty in FC NN
    """

    def __init__(self, trainings_count: int, fit_using_hps: Optional[HyperparamsEmbedding] = None, fit_using_dataset_stats: bool = False, fit_using_loss_curves: bool = False):
        """
        Args:
            - trainings_count: Number of training which will be performed to predict generalization capability of a model (trainings differs between each other in scale either by having different trainset sizes (across subsets) or different model capacities (across model capacities))
            - fit_using_hps:
            - fit_using_dataset_stats:
            - fit_using_loss_curves:
        """
        self._fit_using_hps = fit_using_hps
        self._fit_using_dataset_stats = fit_using_dataset_stats
        self._fit_using_loss_curves = fit_using_loss_curves
        # We initialize (α, eps0, c∞, η, β, b) parameters to regression results over CIDAR10 dataset with a ResNet architecture (values resulting from https://arxiv.org/pdf/1909.12673.pdf experiments)
        self._leastsquares_params = np.array([0.66, 0.9, 7.14e-14, 19.77, 0.53, 5.87e-2])
        self._trainings_count = trainings_count

        self._use_additional_nn_model = any((fit_using_hps, fit_using_dataset_stats, fit_using_loss_curves))
        if self._use_additional_nn_model:
            # Define additional meta model which, combined with previous lightweight regression model, predicts generalization capability of a model over full dataset
            # Default NN input data: Best train loss, best valid loss, trainset size and model size for each trainset subsets (4 * training_count) + leastsquares fitted model's parameter vector (self._leastsquares_params) and prediction (+1) + full dataset size, model parameter count (+2)
            self._input_size = 4 * len(self._trainings_count) + len(self._leastsquares_params) + 3  # TODO: change NN input to be constant sized (not depending on subset count)?
            if self._fit_using_hps is not None:
                self._input_size += self._fit_using_hps.embedding_size  # + Input hyperparameter dict embedding/distance-like-hash to NN
            if self._fit_using_dataset_stats:
                self._input_size += ...  # + Input dataset stats like trainset/validset ratio, data shape, batch_size, mean;variance;quartiles;... of trainset targets, data-type #TODO: see https://arxiv.org/pdf/1810.06305.pdf for interesting dataset embedding/stats
            if self._fit_using_loss_curves:
                self._input_size += ...  # + Validation and training losses evaluated during training iterations
            linear1 = nn.Linear(in_features=self._input_size, out_features=64)
            linear2 = nn.Linear(in_features=64, out_features=2)  # Outputs valid and train losses
            self._nn_metamodel = nn.Sequential(linear1, nn.Tanh(), linear2, nn.Tanh())
            nn.init.xavier_uniform_(self._nn_metamodel.weight.data, gain=nn.init.calculate_gain('tanh'))
            self._nn_metamodel.bias.data.fill_(0.)

    @staticmethod
    def error_landscape_estimation(metaparms: np.array, m: Optional[int] = None, n: Optional[int] = None) -> float:
        """ Enveloppe function modeling how best valid loss varies according to model size (m) and trainset size (n).
        This simple model's parameters (metaparms) can be fitted on a few training results over trainset subsets (~5-6 subsets) in order to predict model's generealization capability over full trainset without having to train on whole dataset.
        Thus, fitting this model during hyperparameter search can save time by estimating how promising is a hyperparmeter setup.
        NOTE: if 'm' is None, model capacity is considered to be constant and 'b'/'beta' term will be zeroed/ignored; if 'n' is None, then trainset size is considered to be constant and 'a'/'alpha' term will be zeroed/ignored (allows to simplify model, as constant terms in 'emn' are redundant with 'cinf' parameter)
        TODO: make bayesian estimate of least-squares regression uncertainty of this model by choosing appropriate à-prioris (+ combine this estimation with NN's uncertainty)
        """
        eps0, cinf, eta = metaparms[0 if n is None else 1: 3]
        emn = cinf
        if n is not None:
            a, alpha = 1., metaparms[0]  # 'a=1' because it is a redundant parameter: equivalent to divide 'emn' by 'a' with 'eta' value replaced by 'a * eta'
            emn += a * np.power(float(n), -alpha)
        if m is not None:
            b, beta = metaparms[-2:]
            emn += b * np.power(float(m), -beta)
        return eps0 * np.absolute(emn / (emn - eta * 1j))  # Complex absolute, i.e. 2D L2 norm

    def fit_generalization(self, trainsets: Sequence[DataLoader], models: Sequence[nn.Module], best_valid_losses: Sequence[Union[float, torch.FloatTensor]], best_train_losses: Optional[Sequence[Union[float, torch.FloatTensor]]] = None):
        model_capacities = [deepcv.meta.nn.get_model_capacity(m) for m in models]
        cst_modelsize = deepcv.utils.is_roughtly_constant(model_capacities)
        if cst_modelsize:
            # If model capacity doesn't change, we can simplify model regression by removing 'b' and 'beta' parameters (constant term which can be modeled by 'cinf' parameter)
            params = self._leastsquares_params[: -2]

        trainset_sizes = [len(dl.dataset) for dl in trainsets]
        cst_datasize = deepcv.utils.is_roughtly_constant(trainset_sizes)
        if cst_datasize:
            # If traiset size doesn't change across results, we can simplify model regression by removing 'alpha' parameter (constant term which can be modeled by 'cinf' parameter)
            params = params[1:]

        # Fit basic `error_landscape_estimation` model over subsets training results using least squares regression of error estimates divergence
        def _error_landscape_divergence(metaparms: np.array) -> float:
            preds = [GeneralizationAcrossScalesPredictor.error_landscape_estimation(metaparms, None if cst_modelsize else m, None if cst_datasize else n)
                     for m, n in zip(model_capacities, trainset_sizes)]
            return [(pred - real) / real for pred, real in zip(preds, best_valid_losses)]

        rslt = least_squares(_error_landscape_divergence, x0=params, jac='3-point', bounds=(0., 200), method='dogbox', loss='soft_l1')
        self._leastsquares_params[1 if cst_datasize else None: -2 if cst_modelsize else None] = rslt.x  # TODO: smooth averaging window instead of pure update?

        # Additional online training of fully connected model for better validation error landscape prediction
        if self._use_additional_nn_model:
            raise NotImplementedError
            hhp = {'': ...}
            x = (trainset_sizes, model_capacities, best_valid_losses, best_train_losses)
            loss = torch.optim.RMSprop(self._nn_metamodel.params, lr=hhp['lr'], weight_decay=hhp['weight_decay'], momentum=hhp['momentum'])
            # TODO: basic training procedure
            # TODO: online training considerations
            # TODO: create and train on 'meta' dataset from MLFlow?

    def forward(self, model: Union[nn.Module, int], trainset: Union[DataLoader, int]) -> float:
        model_capacity = deepcv.meta.nn.get_model_capacity(model) if isinstance(model, nn.Module) else model
        trainset_size = trainset if isinstance(trainset, int) else len(trainset)
        estimation = GeneralizationAcrossScalesPredictor.error_landscape_estimation(self._leastsquares_params, model_capacity, trainset_size)

        if self._use_additional_nn_model:
            raise NotImplementedError
            # By default our model takes best train loss, best valid loss, model capacity and dataset size for each dataset subsets along with full dataset size, model capacity and 'error_landscape_estimation' model parameter vector previously fitted with least-squares regression
            x = ...  # torch.Tensor([estimation, *self._leastsquares_params, model_capacity, trainset_size, subsets_results])
            if self._fit_using_hps:
                # TODO: Process hyper-parameters into a distance-like hash/embedding
                x = torch.cat(x, ...)
            if self._fit_using_dataset_stats:
                # TODO: Process trainset stats and feed it to our model
                x = torch.cat(x, ...)
            if self._fit_using_loss_curves:
                # TODO: Append loss curves resulting from trainset and validset evaluations to metamodel input
                x = torch.cat(x, ...)
            # We apply fully connected NN with a residual link from valid error estimate of fitted 'error_landscape_estimation' model to NN's output
            return torch.cat(self._nn_metamodel(x), torch.from_numy(np.array([estimation, 0.])))
        return estimation


def to_hyperparameters(hp: Union[Dict[str, Any], Hyperparameters], defaults: Optional[Union[Dict[str, Any], Hyperparameters]] = None, raise_if_missing: bool = True) -> Union[Hyperparameters, Tuple[Hyperparameters, List[str]]]:
    """ Converts given parameters Dict to a `deepcv.meta.hyperparams.Hyperparameters` object if needed. Alse allows you to check required and default hyperparameter through `defaults` argument (see `deepcv.meta.hyperparams.Hyperparameters.with_defaults` for more details)
    Args:
        - hp: Parameter dict or `deepcv.meta.hyperparams.Hyperparameters` object
        - defaults: Optional argument specifying required and default (hyper)parameter(s) (see `deepcv.meta.hyperparams.Hyperparameters.with_defaults` for more details)
        - raise_if_missing: Boolean indicating whether if this function should raise an exception if `defaults` specifies mandatory (hyper)parameters which are missing in `hp`
    Returns resulting `deepcv.meta.hyperparams.Hyperparameters` object with provided defaults, and eventually also returns missing hyperparameters which are missing according to `defaults` argument, if provided.
    """
    if not isinstance(hp, Hyperparameters):
        hp = Hyperparameters(**hp)
    if defaults is not None:
        hp, missing = hp.with_defaults(defaults)
        if len(missing) > 0:
            msg = f'Error: Missing mandatory (hyper)parameter(s) (missing="{missing}").'
            logging.error(msg)
            if raise_if_missing:
                raise ValueError(msg)
        return hp, missing
    return hp


def merge_hyperparameters(*dicts: Iterable[Dict[str, Any]]) -> Hyperparameters:
    """ Utils function used to merge given dictionnaries into a `hyperparams.Hyperparameters` class instance """
    merged = deepcv.utils.merge_dicts(*dicts)
    return Hyperparameters(*merged)


def hp_search(hp_space: Dict[str, Any], model: nn.Module, training_procedure: Callable, dataloaders: Tuple[DataLoader], pred_across_scales: bool = False, subset_sizes: Sequence[float] = [0.005, 0.015, 0.03, 0.05, 0.07, 0.09]):
    """ NNI Hyperparameter search trial procedure, with optional generalization prediction across scales """
    hp = nni.get_next_parameter()
    logging.info(f'> Hyperparameter search experiment "{nni.get_experiment_id()}" -- trial NO#{nni.get_trial_id()}...\nsampled_hp="{hp}"\n')
    trainset, validset = dataloaders[0], dataloaders[1]

    if pred_across_scales:
        logging.info('Using "prediction of model\'s generalization across scales" technique by performing multiple trainings on small trainset subsets to be able to predict validation error if we had trained model on full trainset.')
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        subsets = [deepcv.meta.data.datasets.get_random_subset_dataloader(
            trainset, ss, batch_size=hp['batch_size'], num_workers=num_workers, pin_memory=True) for ss in subset_sizes]
        generalization_predictor = GeneralizationAcrossScalesPredictor(len(subsets), fit_using_hps=False, fit_using_loss_curves=False, fit_using_dataset_stats=False)

    # TODO: add ignite training handler to report metrics to nni as intermediate results: nni.report_intermediate_result()

    if pred_across_scales:
        best_valid_losses_across_scales = [training_procedure(model, hp, trainset=s, validset=validset) for s in subsets]
        models = [model, ] * len(best_valid_losses_across_scales)  # We always train the same model across different trainset subset sizes
        generalization_predictor.fit_generalization(subsets, models, best_valid_losses_across_scales)
        predicted_score = generalization_predictor(model, trainset)  # TODO: get trainset as parameter
        nni.report_final_result(predicted_score)
    else:
        rslt = training_procedure(model, hp, dataloaders)
        nni.report_final_result(rslt)

    logging.info(f'######## NNI Hyperparameter search trial NO#{nni.get_trial_id()} done! ########')

# TODO: Make usage of nnictl tensorboard during HP searches? (may better handles tensorboard logs along with cleanner tb server starting and stoping)


nni_single_shot_nas_algorithms = {'ENAS': nni.nas.pytorch.enas.EnasTrainer, 'DARTS': nni.nas.pytorch.darts.DartsTrainer, 'P-DARTS': nni.nas.pytorch.pdarts.PdartsTrainer,
                                  'SPOS': nni.nas.pytorch.spos.SposTrainer, 'CDARTS': nni.nas.pytorch.cdarts.CdartsTrainer, 'ProxylessNAS': nni.nas.pytorch.proxylessnas.ProxylessNasTrainer, 'TestNAS': nni.nas.pytorch.textnas.TextNasTrainer}
nni_classic_nas_algorithms = {'PPOTuner', 'RandomTuner'}


def is_nni_run_standalone():
    """ Simple helper function which returns whether NNI is in standalone trial run mode """
    return nni.get_experiment_id() == r'STANDALONE' and nni.get_trial_id() == r'STANDALONE' and nni.get_sequence_id() == 0


def nni_classic_neural_architecture_search_trial(model: torch.nn.Module, training_procedure: Callable[[Dict[str, Dataset], torch.nn.Module, Union[Dict[str, Any], Hyperparameters]], Any], *other_training_args, **other_training_kwargs):
    """ Applies choices among NNI NAS mutable layers and inputs to model architecture and train it using provided training procedure.
    NNI Classic NAS algorithm (either PPO Tuner or Random Tuner) samples a fixed architeture from mutable NN architecture search space.
    NOTE: NNI NAS also supports various single shot neural architecture search algorithms for faster architecture optimization (performed in a single trial instead of one trial per evaluated/possible architecture).
    If this is the first call to NNI Classic NAS API, NNI's `get_and_apply_next_architecture` will first generate JSON search space based on given `model` (NNI dry run mode, for more details, see https://nni.readthedocs.io/en/latest/NAS/NasReference.html#nni.nas.pytorch.classic_nas.get_and_apply_next_architecture)
    Args:
        - model: Model NAS search space to be sampled from and trained. Provided model should be a regular PyTorch module which contains at least one NNI mutable layer(s) and/or mutable input(s).
        - training_procedure: Training procedure of your own used to run a trial. This function should take model to train as its first argument (sampled NN architecture from `model`).
        - *other_training_args: Any positional arguments which should be provided to `training_procedure`, except for the model to train which is already provided by NNI Architecture sampling.
        - **other_training_kwargs: Any other keyword arguments to be provided to `training_procedure`.
    Returns the training results returned by given `training_procedure`. Note that `training_procedure` should at least return a dict-like object with a `best_valid_loss` entry which will be reported to NNI NAS API (float scalar indicating sampled architecture performances).
    """
    nni.nas.pytorch.classic_nas.get_and_apply_next_architecture(model)
    results = training_procedure(model, *other_training_args, **other_training_kwargs)
    nni.report_final_result(results['best_valid_loss'])
    return results


def nni_single_shot_neural_architecture_search(hp: Union[deepcv.meta.hyperparams.Hyperparameters, Dict[str, Any]], model: nn.Module, loss: nn.modules.loss._Loss, datasets: Tuple[Dataset], opt: Type[torch.optim.Optimizer], backend_conf: BackendConfig = BackendConfig(), metrics: Dict[str, Metric] = {}, nas_mutator: nni.nas.pytorch.mutator.Mutator = None) -> Path:
    """ Train model with provided NAS trainer in order to find out the best NN architecture by training a superset NN instead of performing multiple trainings/trials for each/many possible architectures.
    Args:
        - hp:
        - model:
        - loss:
        - datasets:
        - opt:
        - backend_conf:
        - metrics:
        - nas_mutator:
    Returns a pathlib.Path to a JSON file storing the best NN model architecture found by NNI Single-Shot NAS (JSON file storing mutable layer(s)/input(s) choices made in model search space in order to define best fixed architeture found; This file is also logged to mlflow if there is an active run)
    # TODO: convert ignite metrics for NNI NAS trainer usage if needed
    # TODO: reuse code from ignite training for output path and log final architecture as mlflow artifact
    # TODO: Allow resuming an NNI single shot NAS experiment throught 'hp['resume_from']' parameter (if possible using NNI API?)
    # TODO: look for possible hp scheduling and/or control/searches over training HPs like learning rate during single shot NAS (make sure this isn't already handled by underlying NAS algorithms)?
    # TODO: add support for two-way callbacks using deepcv.utils.EventsHandler in a similar way than ignite_training.train (once ignite_training fully support it)
    """

    if mlflow.active_run() is not None:
        mlflow_experiment_name = mlflow.get_experiment(mlflow.active_run().info.experiment_id).name
        mlflow_run = mlflow.active_run().info.run_id
        mlfow_user = mlflow.active_run().info.user_id
        run_info_msg = f'(mlflow_experiment: "{mlflow_experiment_name}", mlflow_run: "{mlflow_run}", mlflow_user: "{mlflow_user}")'
    else:
        run_info_msg = '(No active MLFlow run nor experiment)'
    logging.info(f'Starting Single-Shot Neural Architecture Search (NNI NAS API) training over NN architecture search space {run_info_msg}.')

    TRAINING_HP_DEFAULTS = {'optimizer_opts': ..., 'scheduler': ..., 'epochs': ..., 'output_path': Path.cwd(
    ) / 'data/04_training/', 'log_output_dir_to_mlflow': True, 'log_progress_every_iters': 100, 'seed': None, 'resume_from': '', 'deterministic_cudnn': False}
    hp, _ = deepcv.meta.hyperparams.to_hyperparameters(hp, TRAINING_HP_DEFAULTS, raise_if_missing=True)
    deepcv.utils.setup_cudnn(deterministic=hp['deterministic_cudnn'], seed=backend_conf.rank + hp['seed'])  # In distributed setup, we need to have different seed for each workers
    device = backend_conf.device
    model = model.to(device, non_blocking=True)
    num_workers = max(1, (backend_conf.ncpu - 1) // (backend_conf.ngpus_current_node if backend_conf.ngpus_current_node > 0 and backend_conf.distributed else 1))
    loss = loss.to(device)
    optimizer = opt(model.parameters(), **hp['optimizer_opts'])
    trainset, *validset_testset = *datasets
    output_architecture_filepath = Path(hp['output_path']) / f'single_shot_nas_{}.json'

    model_mutator = ...
    nas_trainer_callbacks = []

    trainer = nni.nas.pytorch.trainer.Trainer(model=model, mutator=model_mutator, loss=loss, metrics=metrics, optimizer=optimizer, num_epochs=hp['epochs'],
                                              trainset=trainset, validset=validset_testset[0], batch_size=hp['batch_size'], num_workers=num_workers,
                                              device=device, log_frequency=hp['log_progress_every_iters'], callbacks=nas_trainer_callbacks)

    # Train model with provided NAS trainer in order to find out the best NN architecture by training a superset NN instead of performing multiple trainings/trials for each/many possible architectures
    trainer.train()
    logging.info(f'Single-shot NAS training done. Validating model architecture... {run_info_msg}')
    trainer.validate()
    logging.info(f'Saving obtained NN architecture from NNI Single-Shot NAS algorithm as a JSON file and logging it to mlfow if possible... {run_info_msg}')
    trainer.export(file=output_architecture_filepath)
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(output_architecture_filepath))
    logging.info(f'Single-Shot NAS trainning procedure completed. {run_info_msg}')


def run_nni_hp_search(pipeline_name, model, backend_conf, hp):
    # TODO: Fill NNI config
    gen_nni_config(...)
    # TODO: Handle NNI NAS Model (Generate HP Space from NNI mutable Layers/inputs in model(s))
    # TODO: merge NNI HP Search space from user with generated NAS Search space
    # TODO: Start NNI process with given pipeline as trial code with nnictl
    cmd = f'nnictl ss-gen-space'


def gen_nni_config(common_nni_config_file: Union[str, Path], kedro_pipeline: str, optimize_mode: str = 'minimize', hp_tunner: str = 'TPE', early_stopping: Optional[str] = 'Medianstop'):
    """ Generates NNI configuration file in order to run an NNI Hyperparameters Search on given pipeline (new/full NNI config file will be save in the same directory as `common_nni_config_file` and will be named after its respective pipeline name).
    Fills missing NNI configuration fields from defaults/commons NNI YAML config (won't change any existing values from given NNI YAML configuration but appends missing parameters with some defaults).
    It means given NNI YAML config file should only contain parameters which are common to any NNI HP/NAS API usage in DeepCV and this function will populate other parameters according to a specific training pipeline.
    NOTE: `gen_nni_config` wont overwrite any existing NNI configuration named after the same pipeline (i.e., if '{kedro_pipeline}_nni_config.yml' already exists, this function wont do anything).
    .. See [NNI HP API documentation for more details on NNI YAML configuration file](https://nni.readthedocs.io/en/latest/hyperparameter_tune.html)
    """
    common_nni_config_file = Path(common_nni_config_file)
    new_config_path = common_nni_config_file.parent / f'{kedro_pipeline}_nni_config.yml'

    if not common_nni_config_file.exists():
        msg = f'Error: Couldn\'t find provided NNI config defaults/template/commons at: "{common_nni_config_file}"'
        logging.error(msg)
        raise FileNotFoundError(msg)
    if new_config_path.exists():
        logging.warn(f'Warning: `deepcv.meta.hyperparams.gen_nni_config` called but YAML NNI config file already exists for this pipeline ("{kedro_pipeline}"), '
                     f'"{new_config_path.name}" YAML config wont be modified, you may want to delete it before if you need to update it.\n'
                     f'Also note that you can customize "{common_nni_config_file}" config if you need to change NNI common behavior for any NNI HP/NAS API usage in DeepCV (Hyperparameter searches and Neural Architecture Searches based on NNI); All NNI configuration are generated from this template/common/default YAML config. See also "deepcv.meta.hyperparams.gen_nni_config" function for more details about NNI config handling in DeepCV.')
        return

    nni_config = anyconfig.load(common_nni_config_file, ac_parser='yaml')

    def _set_parameter_if_not_defined(nni_config, parameter_name: str, default_value: Any):
        nni_config[parameter_name] = getattr(nni_config, parameter_name, default_value)

    _set_parameter_if_not_defined(nni_config, 'authorName', __author__)
    _set_parameter_if_not_defined(nni_config, 'experimentName', nni.get_experiment_id())
    _set_parameter_if_not_defined(nni_config, 'searchSpacePath', common_nni_config_file / r'hp_search_spaces/{pipeline_name}_search_space.json')
    _set_parameter_if_not_defined(nni_config, 'trialConcurrency', 1)
    _set_parameter_if_not_defined(nni_config, 'maxTrialNum', -1)
    _set_parameter_if_not_defined(nni_config, 'trainingServicePlatform', 'local')

    trial_conf = nni_config['trial'] if 'trial' in nni_config else dict()
    _set_parameter_if_not_defined(trial_conf, 'command', f'kedro run --pipeline={kedro_pipeline}')
    _set_parameter_if_not_defined(trial_conf, 'codeDir', common_nni_config_file / r'../../src/deepcv')
    _set_parameter_if_not_defined(trial_conf, 'gpuNum', 0)
    nni_config['trial'] = trial_conf

    tuner_conf = nni_config['tuner'] if 'tuner' in nni_config else dict()
    _set_parameter_if_not_defined(tuner_conf, 'builtinTunerName', hp_tunner)
    _set_parameter_if_not_defined(tuner_conf, 'classArgs', {'optimize_mode': optimize_mode})
    nni_config['tuner'] = tuner_conf

    if early_stopping is not None:
        assesor_conf = nni_config['assessor'] if 'assessor' in nni_config else dict()
        _set_parameter_if_not_defined(assesor_conf, 'builtinAssessorName', early_stopping)
        _set_parameter_if_not_defined(assesor_conf, 'classArgs', {'optimize_mode': optimize_mode, 'start_step': 8})
        nni_config['assessor'] = assesor_conf

    # Save final NNI configuration as a new YAML file named after its respective training pipeline
    anyconfig.dump(nni_config, new_config_path, ac_parser='yaml')


def sample_nni_hp_space(model_hps: Union[Dict[str, Any], Hyperparameters], training_hps: Union[Dict[str, Any], Hyperparameters]) -> Tuple[Union[Dict[str, Any], Hyperparameters], Union[Dict[str, Any], Hyperparameters]]:
    """ Sample hyperparameters from NNI search space and merge those with given model definition and training procedure hyperparameters (which are probably from YAML config) """
    params_from_nni = nni.get_next_parameter()

    # Fill values sampled from NNI seach space in their respective hyperparameters set (model_hps or training_hps)
    for name, value in params_from_nni.items():
        is_model = name.startswith('model:')
        if not is_model and not name.startswith('training:'):
            raise ValueError('Error: NNI hyperparameter names should either start with `training:` or `model:` to specify whether parameter belongs to training procedure or model definition.')

        # Recursive call to dict.__getitem__, which allows to access nested parameters by using a `.` between namespaces
        * hierachy, parameter_name = name.split('.')[1:]
        functools.reduce(dict.__getitem__, [model_hps if is_model else training_hps, *hierachy])[parameter_name] = value

    return model_hps, training_hps


def get_hp_position_in_search_space(hp, hp_search_space):
    # TODO: return hp set position in given search space (aggregates position of each searchable parameters in their respective search range, nested choices/sampling multiplies its parent relative position)
    raise NotImplementedError
    return torch.zeros((1, 10))


def generate_hp_space_template(hyperparams: Union[Dict[str, Any], Hyperparameters], save_filename: str = r'nni_hp_space_template.json', exclude_params: Optional[Sequence[str]] = None, include_params: Optional[Sequence[str]] = None) -> Tuple[Dict[str, Any], Path]:
    """ Generates an hyperparameter space template/draft from given hyperparameter set, making it easier to define your JSON (NNI spec.) hp space definition (generates a start point for your JSON hp space)
    # TODO: active learning of a lightweight model which infers (options, low, high, q, mu and/or sigma) from (parameter name, type, hierarchy/context)
    # TODO: Samples from hp space should be merged to `hp` dict from YAML and vis versa, a given `hp` set should have a position in an hp search space
    """
    options, low, high, q, mu, sigma = list(), 0., 1., 1, 0., 1.  # q is the discrete step size
    nni_search_space_specs = [{'_type': 'choice', '_value': options},
                              # `randint` is equivalent to `quniform` with `q = 1` but may be interpreted as unordered by some NNI hp search tuners while `quniform` is always ordered
                              {'_type': 'randint', '_value': [low, high]},
                              {'_type': 'uniform', '_value': [low, high]},
                              {'_type': 'quniform', '_value': [low, high, q]},
                              {'_type': 'loguniform', '_value': [low, high]},
                              {'_type': 'qloguniform', '_value': [low, high, q]},
                              {'_type': 'normal', '_value': [mu, sigma]},
                              {'_type': 'qnormal', '_value': [mu, sigma, q]},
                              {'_type': 'lognormal', '_value': [mu, sigma]},
                              {'_type': 'qlognormal', '_value': [mu, sigma, q]}]
    search_space_types = {spec['_type'] for spec in nni_search_space_specs}

    exclude_params = set() if exclude_params is None else set(exclude_params)
    include_params = set() if include_params is None else set(include_params)
    hp_search_space = {n: dict(_type=v, _value=[v]) for n, v in hyperparams.items() if n in include_params and n not in exclude_params}
    hp_search_space_json = Path.cwd() / save_filename

    # Save hyperparameters search space template as JSON file
    with open(hp_search_space_json, 'w') as json_file:
        json.dump(hp_search_space, json_file)

    return hp_search_space, hp_search_space_json
    # "dropout_keepprob": {"_type": "uniform", "_value": [0.1, 0.5]},
    # "conv_size": {"_type": "choice", "_value": [2, 3, 5, 7]},
    # "hidden_size": {"_type": "choice", "_value": [124, 512, 1024]},
    # "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128, 256]},
    # "lr": {"_type": "uniform", "_value": [0.0001, 0.1]}
    # }


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
