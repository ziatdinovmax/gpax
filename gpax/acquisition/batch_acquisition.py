"""
batch_acquisition.py
==============

Batch-mode acquisition functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Type, Optional, Callable

import jax.numpy as jnp
from jax import vmap

from ..models.gp import ExactGP
from ..utils import random_sample_dict
from .base_acq import ei, ucb, poi


def compute_batch_acquisition(acquisition_type: Callable,
                              model: Type[ExactGP],
                              X: jnp.ndarray,
                              *acq_args,
                              maximize_distance: bool = False,
                              n_evals: int = 1,
                              subsample_size: int = 1,
                              indices: Optional[jnp.ndarray] = None,
                              **kwargs) -> jnp.ndarray:
    """
    Computes batch-mode acquisition function of a given type
    """
    if model.mcmc is None:
        raise ValueError("The model needs to be fully Bayesian")

    samples = random_sample_dict(model.get_samples(), subsample_size)
    f = vmap(acquisition_type, in_axes=(None, None, 0) + (None,) * len(acq_args))

    if not maximize_distance:
        acq = f(model, X, samples, *acq_args, **kwargs)
    else:
        X_ = jnp.array(indices) if indices is not None else jnp.array(X)
        acq_all, dist_all = [], []

        for _ in range(n_evals):
            acq = f(model, X_, samples, *acq_args, **kwargs)
            points = acq.argmax(-1)
            d = jnp.linalg.norm(points).mean()
            acq_all.append(acq)
            dist_all.append(d)

        idx = jnp.array(dist_all).argmax()
        acq = acq_all[idx]

    return acq


def qEI(model: Type[ExactGP],
        X: jnp.ndarray,
        maximize: bool = False,
        noiseless: bool = False,
        maximize_distance: bool = False,
        n_evals: int = 1,
        subsample_size: int = 1,
        indices: Optional[jnp.ndarray] = None,
        **kwargs) -> jnp.ndarray:
    """
    Batch-mode Expected Improvement
    
    Args:
        model: trained model
        X: new inputs
        maximize: If True, assumes that BO is solving maximization problem
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        maximize_distance:
            Selects a subsample with a maximum distance between acq.argmax() points
        n_evals:
            Number of evaluations (how many times a ramdom subsample is drawn)
            when maximizing distance between maxima of different EIs in a batch.
        subsample_size:
            Size of the subsample from the GP model's MCMC samples.
        indices:
            Indices of the input points.

    Returns:
        The computed batch Expected Improvement values at the provided input points X.
    """

    return compute_batch_acquisition(
        ei, model, X, maximize, noiseless,
        maximize_distance=maximize_distance,
        n_evals=n_evals, subsample_size=subsample_size,
        indices=indices, **kwargs)


def qUCB(model: Type[ExactGP],
         X: jnp.ndarray,
         beta: float = 0.25,
         maximize: bool = False,
         noiseless: bool = False,
         maximize_distance: bool = False,
         n_evals: int = 1,
         subsample_size: int = 1,
         indices: Optional[jnp.ndarray] = None,
         **kwargs) -> jnp.ndarray:
    """
    Batch-mode Upper Confidence Bound
    
    Args:
        model: trained model
        X: new inputs
        beta: the exploration-exploitation trade-off
        maximize: If True, assumes that BO is solving maximization problem
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        maximize_distance:
            Selects a subsample with a maximum distance between acq.argmax() points
        n_evals:
            Number of evaluations (how many times a ramdom subsample is drawn)
            when maximizing distance between maxima of different EIs in a batch.
        subsample_size:
            Size of the subsample from the GP model's MCMC samples.
        indices:
            Indices of the input points.

    Returns:
        The computed batch Upper Confidence Bound values at the provided input points X.
    """

    return compute_batch_acquisition(
        ucb, model, X, beta, maximize, noiseless,
        maximize_distance=maximize_distance,
        n_evals=n_evals, subsample_size=subsample_size,
        indices=indices, **kwargs)


def qPOI(model: Type[ExactGP],
         X: jnp.ndarray,
         xi: float = .001,
         maximize: bool = False,
         noiseless: bool = False,
         maximize_distance: bool = False,
         n_evals: int = 1,
         subsample_size: int = 1,
         indices: Optional[jnp.ndarray] = None,
         **kwargs) -> jnp.ndarray:
    """
    Batch-mode Probability of Improvement

    Args:
        model: trained model
        X: new inputs
        xi: the exploration-exploitation trade-off
        maximize: If True, assumes that BO is solving maximization problem
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        maximize_distance:
            Selects a subsample with a maximum distance between acq.argmax() points
        n_evals:
            Number of evaluations (how many times a ramdom subsample is drawn)
            when maximizing distance between maxima of different EIs in a batch.
        subsample_size:
            Size of the subsample from the GP model's MCMC samples.
        indices:
            Indices of the input points.

    """

    return compute_batch_acquisition(
        poi, model, X, xi, maximize, noiseless,
        maximize_distance=maximize_distance,
        n_evals=n_evals, subsample_size=subsample_size,
        indices=indices, **kwargs)