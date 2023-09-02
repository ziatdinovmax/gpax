"""
batch_acquisition.py
==============

Batch-mode acquisition functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Type, Optional, Callable

import jax.numpy as jnp
from jax import vmap
import jax.random as jra

from ..models.gp import ExactGP
from ..utils import random_sample_dict
from .acquisition import ei, ucb, poi, kg


def _compute_batch_acquisition(
        rng_key: jnp.ndarray,
        model: Type[ExactGP],
        X: jnp.ndarray,
        single_acq_fn: Callable,
        maximize_distance: bool = False,
        subsample_size: int = 1,
        n_evals: int = 10,
        indices: Optional[jnp.ndarray] = None,
        **kwargs) -> jnp.ndarray:
    """Function for computing batch acquisition of a given type"""

    if model.mcmc is None:
        raise ValueError("The model needs to be fully Bayesian")

    X = X[:, None] if X.ndim < 2 else X

    f = vmap(single_acq_fn, in_axes=(0, None))

    if not maximize_distance:
        samples = random_sample_dict(model.get_samples(), subsample_size, rng_key)
        acq = f(samples, X)

    else:
        X_ = jnp.array(indices) if indices is not None else jnp.array(X)

        def compute_acq_and_distance(subkey):
            samples = random_sample_dict(model.get_samples(), subsample_size, subkey)
            acq = f(samples, X_)
            points = acq.argmax(-1)
            d = jnp.linalg.norm(points).mean()
            return acq, d

        subkeys = jra.split(rng_key, num=n_evals)
        acq_all, dist_all = vmap(compute_acq_and_distance)(subkeys)
        idx = dist_all.argmax()
        acq = acq_all[idx]

    return acq


def qEI(rng_key: jnp.ndarray,
        model: Type[ExactGP],
        X: jnp.ndarray,
        best_f: float = None,
        maximize: bool = False,
        noiseless: bool = False,
        maximize_distance: bool = False,
        subsample_size: int = 1,
        n_evals: int = 10,
        indices: Optional[jnp.ndarray] = None,
        **kwargs) -> jnp.ndarray:
    """
    Batch-mode Expected Improvement

    qEI computes the Expected Improvement values for given input points `X` using multiple randomly drawn samples
    from the HMC-inferred model's posterior. If `maximize_distance` is enabled, qEI considers diversity among the
    posterior samples by maximizing the mean distance between samples that give the highest acquisition
    values across multiple evaluations.

    Args:
        rng_key: random number generator key
        model: trained model
        X: new inputs
        best_f:
            Best function value observed so far. Derived from the predictive mean
            when not provided by a user.
        maximize:
            If True, assumes that BO is solving maximization problem
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        maximize_distance:
            If set to True, it means we want our batch to contain points that
            are as far apart as possible in the acquisition function space.
            This encourages diversity in the batch.
        subsample_size:
            Size of the subsample from the GP model's MCMC samples.
        n_evals:
            Number of evaluations (how many times a ramdom subsample is drawn)
            when maximizing distance between maxima of different EIs in a batch.
        indices:
            Indices of the input points.

    Returns:
        The computed batch Expected Improvement values at the provided input points X.
    """

    def single_acq(sample, X):
        mean, cov = model.get_mvn_posterior(X, sample, noiseless, **kwargs)
        return ei((mean, cov.diagonal()), best_f, maximize)

    return _compute_batch_acquisition(
        rng_key, model, X, single_acq, maximize_distance,
        subsample_size, n_evals, indices, **kwargs)


def qUCB(rng_key: jnp.ndarray,
         model: Type[ExactGP],
         X: jnp.ndarray,
         beta: float = 0.25,
         maximize: bool = False,
         noiseless: bool = False,
         maximize_distance: bool = False,
         subsample_size: int = 1,
         n_evals: int = 10,
         indices: Optional[jnp.ndarray] = None,
         **kwargs) -> jnp.ndarray:
    """
    Batch-mode Upper Confidence Bound

    qUCB computes the Upper Confidence Bound values for given input points `X` using multiple randomly drawn samples
    from the HMC-inferred model's posterior. If `maximize_distance` is enabled, qUCB considers diversity among the
    posterior samples by maximizing the mean distance between samples that give the highest acquisition
    values across multiple evaluations.

    Args:
        rng_key: random number generator key
        model: trained model
        X: new inputs
        best_f:
            Best function value observed so far. Derived from the predictive mean
            when not provided by a user.
        maximize:
            If True, assumes that BO is solving maximization problem
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        maximize_distance:
            If set to True, it means we want our batch to contain points that
            are as far apart as possible in the acquisition function space.
            This encourages diversity in the batch.
        subsample_size:
            Size of the subsample from the GP model's MCMC samples.
        n_evals:
            Number of evaluations (how many times a ramdom subsample is drawn)
            when maximizing distance between maxima of different EIs in a batch.
        indices:
            Indices of the input points.

    Returns:
        The computed batch Expected Improvement values at the provided input points X.
    """

    def single_acq(sample, X):
        mean, cov = model.get_mvn_posterior(X, sample, noiseless, **kwargs)
        return ucb((mean, cov.diagonal()), beta, maximize)

    return _compute_batch_acquisition(
        rng_key, model, X, single_acq, maximize_distance,
        subsample_size, n_evals, indices, **kwargs)


def qPOI(rng_key: jnp.ndarray,
         model: Type[ExactGP],
         X: jnp.ndarray,
         best_f: float = None,
         maximize: bool = False,
         noiseless: bool = False,
         maximize_distance: bool = False,
         subsample_size: int = 1,
         n_evals: int = 10,
         indices: Optional[jnp.ndarray] = None,
         **kwargs) -> jnp.ndarray:
    """
    Batch-mode Probability of Improvement

    qPOI computes the Probability of Improvement values for given input points `X` using multiple randomly drawn samples
    from the HMC-inferred model's posterior. If `maximize_distance` is enabled, qPOI considers diversity among the
    posterior samples by maximizing the mean distance between samples that give the highest acquisition
    values across multiple evaluations.

    Args:
        rng_key: random number generator key
        model: trained model
        X: new inputs
        best_f:
            Best function value observed so far. Derived from the predictive mean
            when not provided by a user.
        maximize:
            If True, assumes that BO is solving maximization problem
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        maximize_distance:
            If set to True, it means we want our batch to contain points that
            are as far apart as possible in the acquisition function space.
            This encourages diversity in the batch.
        subsample_size:
            Size of the subsample from the GP model's MCMC samples.
        n_evals:
            Number of evaluations (how many times a ramdom subsample is drawn)
            when maximizing distance between maxima of different EIs in a batch.
        indices:
            Indices of the input points.

    Returns:
        The computed batch Expected Improvement values at the provided input points X.
    """

    def single_acq(sample, X):
        mean, cov = model.get_mvn_posterior(X, sample, noiseless, **kwargs)
        return poi((mean, cov.diagonal()), best_f, maximize)

    return _compute_batch_acquisition(
        rng_key, model, X, single_acq, maximize_distance,
        subsample_size, n_evals, indices, **kwargs)


def qKG(rng_key: jnp.ndarray,
        model: Type[ExactGP],
        X: jnp.ndarray,
        n: int = 10,
        maximize: bool = False,
        noiseless: bool = False,
        maximize_distance: bool = False,
        subsample_size: int = 1,
        n_evals: int = 10,
        indices: Optional[jnp.ndarray] = None,
        **kwargs) -> jnp.ndarray:
    """
    Batch-mode Knowledge Gradient

    qKG computes the Knowledge Gradient values for given input points `X` using multiple randomly drawn samples
    from the HMC-inferred model's posterior. If `maximize_distance` is enabled, qKG considers diversity among the
    posterior samples by maximizing the mean distance between samples that give the highest acquisition
    values across multiple evaluations.

    Args:
        rng_key: random number generator key
        model: trained model
        X: new inputs
        n: number of simulated samples for each point in X
        maximize: If True, assumes that BO is solving maximization problem
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        maximize_distance:
            If set to True, it means we want our batch to contain points that
            are as far apart as possible in the acquisition function space.
            This encourages diversity in the batch.
        subsample_size:
            Size of the subsample from the GP model's MCMC samples.
        n_evals:
            Number of evaluations (how many times a ramdom subsample is drawn)
            when maximizing distance between maxima of different EIs in a batch.
        indices:
            Indices of the input points.

    Returns:
        The computed batch Knowledge Gradient values at the provided input points X.
    """
    def single_acq(sample, X):
        return kg(model, X, sample, rng_key, n, maximize, noiseless, **kwargs)

    return _compute_batch_acquisition(
        rng_key, model, X, single_acq, maximize_distance,
        subsample_size, n_evals, indices, **kwargs)
