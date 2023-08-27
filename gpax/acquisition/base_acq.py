"""
base_acq.py
==============

Base acquisition functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Type, Dict, Optional

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from ..models.gp import ExactGP
from ..utils import get_keys


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


def kg(model: Type[ExactGP],
       X_new: jnp.ndarray,
       sample: Dict[str, jnp.ndarray],
       n: int = 10,
       maximize: bool = True,
       noiseless: bool = True,
       rng_key: Optional[jnp.ndarray] = None,
       **kwargs):
    
    r"""
    Knowledge gradient

    Args:
        model: trained model
        X: new inputs with shape (N, D), where D is a feature dimension
        sample: a single sample with model parameters
        n: Number fo simulated samples (Defaults to 10)
        maximize: If True, assumes that BO is solving maximization problem
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        rng_key: random number generator key
        **jitter:
            Small positive term added to the diagonal part of a covariance
            matrix for numerical stability (Default: 1e-6)
    """

    if rng_key is None:
        rng_key = get_keys()[0]
    if not isinstance(sample, (tuple, list)):
        sample = (sample,)

    X_train_o = model.X_train.copy()
    y_train_o = model.y_train.copy()

    def kg_for_one_point(x_aug, y_aug, mean_o):
        # Update GP model with augmented data (as if y_sim was an actual observation at x)
        model._set_training_data(x_aug, y_aug)
        # Re-evaluate posterior predictive distribution on all the candidate ("test") points
        mean_aug, _ = model.get_mvn_posterior(X_new, *sample, noiseless=noiseless, **kwargs)
        # Find the maximum mean value
        y_fant = mean_aug.max() if maximize else mean_aug.min()
        # Compute adn return the improvement compared to the original maximum mean value
        mean_o_best = mean_o.max() if maximize else mean_o.min()
        u = y_fant - mean_o_best
        if not maximize:
            u = -u
        return u

    # Get posterior distribution for candidate points
    mean, cov = model.get_mvn_posterior(X_new, *sample, noiseless=noiseless, **kwargs)
    # Simulate potential observations
    y_sim = dist.MultivariateNormal(mean, cov).sample(rng_key, sample_shape=(n,))
    # Augment training data with simulated observations
    X_train_aug = jnp.array([jnp.concatenate([X_train_o, x[None]], axis=0) for x in X_new])
    y_train_aug = []
    for ys in y_sim:
        y_train_aug.append(jnp.array([jnp.concatenate([y_train_o, y[None]]) for y in ys]))
    y_train_aug = jnp.array(y_train_aug)
    # Compute KG
    vectorized_kg = jax.vmap(jax.vmap(kg_for_one_point, in_axes=(0, 0, None)), in_axes=(None, 0, None))
    kg_values = vectorized_kg(X_train_aug, y_train_aug, mean)

    # Reset training data to the original
    model._set_training_data(X_train_o, y_train_o)

    return kg_values.mean(0)
