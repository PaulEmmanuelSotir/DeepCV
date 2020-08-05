#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Hyperparameter search meta module - hyperparams.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir

# To-Do List
    - TODO: For hyperparameter embedding: read about graph embedding techniques like: https://github.com/google-research/google-research/tree/master/graph_embedding/ddgk and https://github.com/google-research/google-research/tree/master/graph_embedding/watch_your_step
"""
import logging
from typing import Sequence, Iterable, Callable, Dict, Tuple, Any, Union, Optional, List, Type

import mlflow
import networkx
import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import deepcv.utils
import deepcv.meta.nn
import deepcv.meta.data.training_metadata
from deepcv.meta.types_aliases import *

__all__ = ['Hyperparameters', 'HyperparameterSpace', 'HyperparamsEmbedding', 'GeneralizationAcrossScalesPredictor', 'to_hyperparameters', 'merge_hyperparameters']
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

    def fit_generalization(self, trainsets: Sequence[DataLoader], models: Sequence[nn.Module], best_valid_losses: Sequence[FLOAT_OR_FLOAT_TENSOR_T], best_train_losses: Optional[Sequence[FLOAT_OR_FLOAT_TENSOR_T]] = None):
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

        rslt = scipy.optimize.least_squares(_error_landscape_divergence, x0=params, jac='3-point', bounds=(0., 200), method='dogbox', loss='soft_l1')
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


def to_hyperparameters(hp: HYPERPARAMS_T, defaults: Optional[HYPERPARAMS_T] = None, raise_if_missing: bool = True) -> Union[Hyperparameters, Tuple[Hyperparameters, List[str]]]:
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

#_______________________________________________ HYPERPARAMS UNIT TESTS _______________________________________________#


if __name__ == '__main__':
    cli = deepcv.utils.import_tests().test_module_cli(__file__)
    cli()
