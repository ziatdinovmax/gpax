"""
base_acq.py
==============

Base acquisition functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Type, Dict

import jax.numpy as jnp
import numpyro.distributions as dist

from ..models.gp import ExactGP


def ei(model: Type[ExactGP],
       X: jnp.ndarray,
       sample: Dict[str, jnp.ndarray],
       maximize: bool = False,
       noiseless: bool = False,
       **kwargs) -> jnp.ndarray:
    r"""
    Expected Improvement

    Args:
        model: trained model
        X: new inputs with shape (N, D), where D is a feature dimension
        sample: a single sample with model parameters
        maximize: If True, assumes that BO is solving maximization problem
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        **jitter:
            Small positive term added to the diagonal part of a covariance
            matrix for numerical stability (Default: 1e-6)
    """
    if not isinstance(sample, (tuple, list)):
        sample = (sample,)
    # Get predictive mean and covariance for a single sample with kernel parameters
    pred, cov = model.get_mvn_posterior(X, *sample, noiseless, **kwargs)
    # Compute standard deviation
    sigma = jnp.sqrt(cov.diagonal())
    # Standard EI computation
    best_f = pred.max() if maximize else pred.min()
    u = (pred - best_f) / sigma
    if not maximize:
        u = -u
    normal = dist.Normal(jnp.zeros_like(u), jnp.ones_like(u))
    ucdf = normal.cdf(u)
    updf = jnp.exp(normal.log_prob(u))
    acq = sigma * (updf + u * ucdf)
    return acq


def ucb(model: Type[ExactGP],
        X: jnp.ndarray,
        sample: Dict[str, jnp.ndarray],
        beta: float = 0.25,
        maximize: bool = False,
        noiseless: bool = False,
        **kwargs) -> jnp.ndarray:
    r"""
    Upper confidence bound

    Args:
        model: trained model
        X: new inputs with shape (N, D), where D is a feature dimension
        sample: a single sample with model parameters
        beta: coefficient balancing exploration-exploitation trade-off
        maximize: If True, assumes that BO is solving maximization problem
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        **jitter:
            Small positive term added to the diagonal part of a covariance
            matrix for numerical stability (Default: 1e-6)
    """
    if not isinstance(sample, (tuple, list)):
        sample = (sample,)
    # Get predictive mean and covariance for a single sample with kernel parameters
    mean, cov = model.get_mvn_posterior(X, *sample, noiseless, **kwargs)
    var = cov.diagonal()
    # Standard UCB derivation
    delta = jnp.sqrt(beta * var)
    if maximize:
        acq = mean + delta
    else:
        acq = delta - mean  # we return a negative acq for argmax in BO
    return acq


def ue(model: Type[ExactGP],
       X: jnp.ndarray,
       sample: Dict[str, jnp.ndarray],
       noiseless: bool = False,
       **kwargs) -> jnp.ndarray:
    r"""
    Uncertainty-based exploration

    Args:
        model: trained model
        X: new inputs with shape (N, D), where D is a feature dimension
        sample: a single sample with model parameters
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        **jitter:
            Small positive term added to the diagonal part of a covariance
            matrix for numerical stability (Default: 1e-6)
    """
    if not isinstance(sample, (tuple, list)):
        sample = (sample,)
    # Get covariance for a single sample with kernel parameters
    _, cov = model.get_mvn_posterior(X, *sample, noiseless, **kwargs)
    # Return variance
    return cov.diagonal()


def poi(model: Type[ExactGP],
        X: jnp.ndarray,
        sample: Dict[str, jnp.ndarray],
        xi: float = 0.01,
        maximize: bool = False,
        noiseless: bool = False,
        **kwargs) -> jnp.ndarray:
    r"""
    Probability of Improvement

    Args:
        model: trained model
        X: new inputs with shape (N, D), where D is a feature dimension
        sample: a single sample with model parameters
        xi: Exploration-exploitation trade-off parameter (Defaults to 0.01)
        maximize: If True, assumes that BO is solving maximization problem
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        **jitter:
            Small positive term added to the diagonal part of a covariance
            matrix for numerical stability (Default: 1e-6)
    """
    if not isinstance(sample, (tuple, list)):
        sample = (sample,)
    # Get predictive mean and covariance for a single sample with kernel parameters
    pred, cov = model.get_mvn_posterior(X, *sample, noiseless, **kwargs)
    # Compute standard deviation
    sigma = jnp.sqrt(cov.diagonal())
    # Standard computation of poi
    best_f = pred.max() if maximize else pred.min()
    u = (pred - best_f - xi) / sigma
    if not maximize:
        u = -u
    normal = dist.Normal(jnp.zeros_like(u), jnp.ones_like(u))
    return normal.cdf(u)
