"""
acquisition.py
==============

Acquisition functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Type, Tuple, Optional

import jax.numpy as jnp
import jax.random as jra
import numpy as onp
import numpyro.distributions as dist

from ..models.gp import ExactGP
from .penalties import compute_penalty


def EI(rng_key: jnp.ndarray, model: Type[ExactGP],
       X: jnp.ndarray,
       maximize: bool = False, n: int = 1,
       noiseless: bool = False,
       penalty: Optional[str] = None,
       recent_points: jnp.ndarray = None,
       grid_indices: jnp.ndarray = None,
       penalty_factor: float = 1.0,
       **kwargs) -> jnp.ndarray:
    r"""
    Expected Improvement

    Args:
        rng_key: JAX random number generator key
        model: trained model
        X: new inputs
        maximize: If True, assumes that BO is solving maximization problem
        n: number of samples drawn from each MVN distribution
           (number of distributions is equal to the number of HMC samples)
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        penalty:
            Penalty applied to the acquisition function to discourage re-evaluation
            at or near points that were recently evaluated. Options are:

            - 'delta':
            The infinite penalty is applied to the recently visited points.

            - 'inverse_distance':
            Modifies the acquisition function by penalizing points near the recent points.

            For the 'inverse_distance', the acqusition function is penalized as:

            .. math::
                \alpha - \lambda \cdot \pi(X, r)

            where :math:`\pi(X, r)` computes a penalty for points in :math:`X` based on their distance to recent points :math:`r`,
            :math:`\alpha` represents the acquisition function, and :math:`\lambda` represents the penalty factor.
        recent_points:
            An array of recently visited points [oldest, ..., newest] provided by user
        grid_indices:
            Grid indices of data points in X array for the penalty term calculation.
            For example, if each data point is an image patch, the indices could correspond
            to the (i, j) pixel coordinates of their centers in the original image.
        penalty_factor:
            Penalty factor :math:`\lambda` in :math:`\alpha - \lambda \cdot \pi(X, r)`
        **jitter:
            Small positive term added to the diagonal part of a covariance
            matrix for numerical stability (Default: 1e-6)
    """
    if penalty and not isinstance(recent_points, (onp.ndarray, jnp.ndarray)):
        raise ValueError("Please provide an array of recently visited points")
    X = X[:, None] if X.ndim < 2 else X
    if model.mcmc is not None:
        y_mean, y_sampled = model.predict(
            rng_key, X, n=n, noiseless=noiseless, **kwargs)
        y_sampled = y_sampled.reshape(n * y_sampled.shape[0], -1)
        mean, sigma = y_sampled.mean(0), y_sampled.std(0)
        best_f = y_mean.max() if maximize else y_mean.min()
    else:
        mean, var = model.predict(
            rng_key, X, noiseless=noiseless, **kwargs)
        sigma = jnp.sqrt(var)
        best_f = mean.max() if maximize else mean.min()
    u = (mean - best_f) / sigma
    if not maximize:
        u = -u
    normal = dist.Normal(jnp.zeros_like(u), jnp.ones_like(u))
    ucdf = normal.cdf(u)
    updf = jnp.exp(normal.log_prob(u))
    acq = sigma * (updf + u * ucdf)
    if penalty:
        X_ = grid_indices if grid_indices is not None else X
        penalties = compute_penalty(X_, recent_points, penalty, penalty_factor)
        acq -= penalties
    return acq


def UCB(rng_key: jnp.ndarray, model: Type[ExactGP],
        X: jnp.ndarray, beta: float = .25,
        maximize: bool = False, n: int = 1,
        noiseless: bool = False,
        penalty: Optional[str] = None,
        recent_points: jnp.ndarray = None,
        grid_indices: jnp.ndarray = None,
        penalty_factor: float = 1.0,
        **kwargs) -> jnp.ndarray:
    r"""
    Upper confidence bound

    Args:
        rng_key: JAX random number generator key
        model: trained model
        X: new inputs
        beta: coefficient balancing exploration-exploitation trade-off
        maximize: If True, assumes that BO is solving maximization problem
        n: number of samples drawn from each MVN distribution
           (number of distributions is equal to the number of HMC samples)
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        penalty:
            Penalty applied to the acquisition function to discourage re-evaluation
            at or near points that were recently evaluated. Options are:

            - 'delta':
            The infinite penalty is applied to the recently visited points.

            - 'inverse_distance':
            Modifies the acquisition function by penalizing points near the recent points.

            For the 'inverse_distance', the acqusition function is penalized as:

            .. math::
                \alpha - \lambda \cdot \pi(X, r)

            where :math:`\pi(X, r)` computes a penalty for points in :math:`X` based on their distance to recent points :math:`r`,
            :math:`\alpha` represents the acquisition function, and :math:`\lambda` represents the penalty factor.
        recent_points:
            An array of recently visited points [oldest, ..., newest] provided by user
        grid_indices:
            Grid indices of data points in X array for the penalty term calculation.
            For example, if each data point is an image patch, the indices could correspond
            to the (i, j) pixel coordinates of their centers in the original image.
        penalty_factor:
            Penalty factor :math:`\lambda` in :math:`\alpha - \lambda \cdot \pi(X, r)`
        **jitter:
            Small positive term added to the diagonal part of a covariance
            matrix for numerical stability (Default: 1e-6)
    """
    if penalty and not isinstance(recent_points, (onp.ndarray, jnp.ndarray)):
        raise ValueError("Please provide an array of recently visited points")
    X = X[:, None] if X.ndim < 2 else X
    if model.mcmc is not None:
        _, y_sampled = model.predict(
            rng_key, X, n=n, noiseless=noiseless, **kwargs)
        y_sampled = y_sampled.reshape(n * y_sampled.shape[0], -1)
        mean, var = y_sampled.mean(0), y_sampled.var(0)
    else:
        mean, var = model.predict(
            rng_key, X, noiseless=noiseless, **kwargs)
    delta = jnp.sqrt(beta * var)
    if maximize:
        acq = mean + delta
    else:
        acq = delta - mean  # we return a negative acq for argmax in BO
    if penalty:
        X_ = grid_indices if grid_indices is not None else X
        penalties = compute_penalty(X_, recent_points, penalty, penalty_factor)
        acq -= penalties
    return acq


def UE(rng_key: jnp.ndarray,
       model: Type[ExactGP],
       X: jnp.ndarray, n: int = 1,
       noiseless: bool = False,
       penalty: Optional[str] = None,
       recent_points: jnp.ndarray = None,
       grid_indices: jnp.ndarray = None,
       penalty_factor: float = 1.0,
       **kwargs) -> jnp.ndarray:
    r"""
    Uncertainty-based exploration

    Args:
        rng_key: JAX random number generator key
        model: trained model
        X: new inputs
        n: number of samples drawn from each MVN distribution
           (number of distributions is equal to the number of HMC samples)
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        penalty:
            Penalty applied to the acquisition function to discourage re-evaluation
            at or near points that were recently evaluated. Options are:

            - 'delta':
            The infinite penalty is applied to the recently visited points.

            - 'inverse_distance':
            Modifies the acquisition function by penalizing points near the recent points.

            For the 'inverse_distance', the acqusition function is penalized as:

            .. math::
                \alpha - \lambda \cdot \pi(X, r)

            where :math:`\pi(X, r)` computes a penalty for points in :math:`X` based on their distance to recent points :math:`r`,
            :math:`\alpha` represents the acquisition function, and :math:`\lambda` represents the penalty factor.
        recent_points:
            An array of recently visited points [oldest, ..., newest] provided by user
        grid_indices:
            Grid indices of data points in X array for the penalty term calculation.
            For example, if each data point is an image patch, the indices could correspond
            to the (i, j) pixel coordinates of their centers in the original image.
        penalty_factor:
            Penalty factor :math:`\lambda` in :math:`\alpha - \lambda \cdot \pi(X, r)`
        **jitter:
            Small positive term added to the diagonal part of a covariance
            matrix for numerical stability (Default: 1e-6)
    """
    if penalty and not isinstance(recent_points, (onp.ndarray, jnp.ndarray)):
        raise ValueError("Please provide an array of recently visited points")
    X = X[:, None] if X.ndim < 2 else X
    if model.mcmc is not None:
        _, y_sampled = model.predict(
            rng_key, X, n=n, noiseless=noiseless, **kwargs)
        y_sampled = y_sampled.mean(1)
        var = y_sampled.var(0)
    else:
        _, var = model.predict(
            rng_key, X, noiseless=noiseless, **kwargs)
    if penalty:
        X_ = grid_indices if grid_indices is not None else X
        penalties = compute_penalty(X_, recent_points, penalty, penalty_factor)
        var -= penalties
    return var


def Thompson(rng_key: jnp.ndarray,
             model: Type[ExactGP],
             X: jnp.ndarray, n: int = 1,
             noiseless: bool = False,
             **kwargs) -> jnp.ndarray:
    """
    Thompson sampling

    Args:
        rng_key: JAX random number generator key
        model: trained model
        X: new inputs
        n: number of samples drawn from the randomly selected MVN distribution
        noiseless:
            Noise-free prediction. It is set to False by default as new/unseen data is assumed
            to follow the same distribution as the training data. Hence, since we introduce a model noise
            for the training data, we also want to include that noise in our prediction.
        **jitter:
            Small positive term added to the diagonal part of a covariance
            matrix for numerical stability (Default: 1e-6)
    """
    if model.mcmc is not None:
        posterior_samples = model.get_samples()
        idx = jra.randint(rng_key, (1,), 0, len(posterior_samples["k_length"]))
        samples = {k: v[idx] for (k, v) in posterior_samples.items()}
        _, tsample = model.predict(
            rng_key, X, samples, n, noiseless=noiseless, **kwargs)
        if n > 1:
            tsample = tsample.mean(1).squeeze()
    else:
        _, tsample = model.sample_from_posterior(
            rng_key, X, n=1, noiseless=noiseless, **kwargs)
    return tsample


def qUCB(rng_key: jnp.ndarray, model: Type[ExactGP],
         X: jnp.ndarray, indices: Optional[jnp.ndarray] = None,
         qbatch_size: int = 4, alpha: float = 1.0, beta: float = .25,
         maximize: bool = True, n: int = 500,
         n_restarts: int = 20, noiseless: bool = False,
         **kwargs) -> jnp.ndarray:
    """
    The acquisition function defined as alpha * mu + sqrt(beta) * sigma
    that can output a "batch" of next points to evaluate. It takes advantage of
    the fact that in MCMC-based GP or DKL we obtain a separate multivariate
    normal posterior for each set of sampled kernel hyperparameters.

    Args:
        rng_key: random number generator key
        model: ExactGP or DKL type of model
        X: input array
        indices: indices of data points in X array. For example, if
            each data point is an image patch, the indices should
            correspond to their (x, y) coordinates in the original image.
        qbatch_size: desired number of sampled points (default: 4)
        alpha: coefficient before mean prediction term (default: 1.0)
        beta: coefficient before variance term (default: 0.25)
        maximize: sign of variance term (+/- if True/False)
        n: number of draws from each multivariate normal posterior
        n_restarts: number of restarts to find a batch of maximally
            separated points to evaluate next
        noiseless: noise-free prediction for new/test data (default: False)

    Returns:
        Computed acquisition function with qbatch x features
        or task x qbatch x features dimensions
    """
    if model.mcmc is None:
        raise NotImplementedError(
            "Currently supports only ExactGP and DKL with MCMC inference")
    dist_all, obj_all = [], []
    X_ = jnp.array(indices) if indices is not None else jnp.array(X)
    for _ in range(n_restarts):
        y_sampled = obtain_samples(
            rng_key, model, X, qbatch_size, n, noiseless, **kwargs)
        mean, var = y_sampled.mean(1), y_sampled.var(1)
        delta = jnp.sqrt(beta * var)
        if maximize:
            obj = alpha * mean + delta
            points = X_[obj.argmax(-1)]
        else:
            obj = alpha * mean - delta
            points = X_[obj.argmin(-1)]
        d = jnp.linalg.norm(points, axis=-1).mean(0)
        dist_all.append(d)
        obj_all.append(obj)
    idx = jnp.array(dist_all).argmax(0)
    if idx.ndim > 0:
        obj_all = jnp.array(obj_all)
        return jnp.array([obj_all[j,:,i] for i, j in enumerate(idx)])
    return obj_all[idx]


def obtain_samples(rng_key: jnp.ndarray, model: Type[ExactGP],
                   X: jnp.ndarray, qbatch_size: int = 4,
                   n: int = 500, noiseless: bool = False,
                   **kwargs) -> jnp.ndarray:
    xbatch_size = kwargs.get("xbatch_size", 100)
    posterior_samples = model.get_samples()
    idx = onp.arange(0, len(posterior_samples["k_length"]))
    onp.random.shuffle(idx)
    idx = idx[:qbatch_size]
    samples = {k: v[idx] for (k, v) in posterior_samples.items()}
    if X.shape[0] > xbatch_size:
        _, y_sampled = model.predict(
            rng_key, X, samples, n,
            noiseless=noiseless, **kwargs)
    else:
        _, y_sampled = model.predict_in_batches(
            rng_key, X, xbatch_size, samples, n,
            noiseless=noiseless, **kwargs)
    return y_sampled
