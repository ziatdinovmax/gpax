"""
acquisition.py
==============

Acquisition functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Type, Optional, Tuple

import jax.numpy as jnp
import jax.random as jra
from jax import vmap
import numpy as onp

from ..models.gp import ExactGP
from .base_acq import ei, ucb, poi, ue, kg
from .penalties import compute_penalty


def _compute_mean_and_var(
        rng_key: jnp.ndarray, model: Type[ExactGP], X: jnp.ndarray,
        n: int, noiseless: bool, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes predictive mean and variance
    """
    if model.mcmc is not None:
        _, y_sampled = model.predict(
            rng_key, X, n=n, noiseless=noiseless, **kwargs)
        y_sampled = y_sampled.reshape(n * y_sampled.shape[0], -1)
        mean, var = y_sampled.mean(0), y_sampled.var(0)
    else:
        mean, var = model.predict(rng_key, X, noiseless=noiseless, **kwargs)
    return mean, var


def _compute_penalties(
        X: jnp.ndarray, recent_points: jnp.ndarray, penalty: str,
        penalty_factor: float, grid_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Computes penaltes for recent points to be substracted
    from acqusition function values
    """
    X_ = grid_indices if grid_indices is not None else X
    return compute_penalty(X_, recent_points, penalty, penalty_factor)


def EI(rng_key: jnp.ndarray, model: Type[ExactGP],
       X: jnp.ndarray, best_f: float = None,
       maximize: bool = False, n: int = 1,
       noiseless: bool = False,
       penalty: Optional[str] = None,
       recent_points: jnp.ndarray = None,
       grid_indices: jnp.ndarray = None,
       penalty_factor: float = 1.0,
       **kwargs) -> jnp.ndarray:
    r"""
    Expected Improvement

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Expected Improvement at an input point :math:`x` is defined as:

    .. math::
        EI(x) =
        \begin{cases}
        (\mu(x) - f^+) \Phi(Z) + \sigma(x) \phi(Z) & \text{if } \sigma(x) > 0 \\
        0 & \text{if } \sigma(x) = 0
        \end{cases}

    where :math:`\mu(x)` is the predictive mean, :math:`\sigma(x)` is the predictive standard deviation,
    :math:`f^+` is the value of the best observed sample. :math:`Z` is defined as:

    .. math::

        Z = \frac{\mu(x) - f^+}{\sigma(x)}

    provided :math:`\sigma(x) > 0`.

    In the case of HMC, the function leverages multiple predictive posteriors, each associated
    with a different HMC sample of the GP model parameters, to capture both prediction uncertainty
    and hyperparameter uncertainty. In this setup, the uncertainty in parameters of probabilistic
    mean function (if any) also contributes to the acquisition function values.

    Args:
        rng_key: JAX random number generator key
        model: trained model
        X: new inputs
        best_f:
            Best function value observed so far. Derived from the predictive mean
            when not provided by a user.
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

    moments = _compute_mean_and_var(rng_key, model, X, n, noiseless, **kwargs)

    acq = ei(moments, best_f, maximize)

    if penalty:
        acq -= _compute_penalties(X, recent_points, penalty, penalty_factor, grid_indices)

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

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Upper Confidence Bound at an input point :math:`x` is defined as:

    .. math::

        UCB(x) = \mu(x) + \kappa \sigma(x)

    where :math:`\mu(x)` is the predictive mean, :math:`\sigma(x)` is the predictive standard deviation,
    and :math:`\kappa` is the exploration-exploitation trade-off parameter.

    In the case of HMC, the function leverages multiple predictive posteriors, each associated
    with a different HMC sample of the GP model parameters, to capture both prediction uncertainty
    and hyperparameter uncertainty. In this setup, the uncertainty in parameters of probabilistic
    mean function (if any) also contributes to the acquisition function values.

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

    moments = _compute_mean_and_var(rng_key, model, X, n, noiseless, **kwargs)

    acq = ucb(moments, beta, maximize)

    if penalty:
        acq -= _compute_penalties(X, recent_points, penalty, penalty_factor, grid_indices)

    return acq


def POI(rng_key: jnp.ndarray, model: Type[ExactGP],
        X: jnp.ndarray, best_f: float = None,
        xi: float = 0.01, maximize: bool = False,
        n: int = 1, noiseless: bool = False,
        penalty: Optional[str] = None,
        recent_points: jnp.ndarray = None,
        grid_indices: jnp.ndarray = None,
        penalty_factor: float = 1.0,
        **kwargs) -> jnp.ndarray:
    r"""
    Probability of Improvement

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Probability of Improvement at an input point :math:`x` is defined as:

    .. math::

        PI(x) = \Phi\left(\frac{\mu(x) - f^+ - \xi}{\sigma(x)}\right)

    where :math:`\mu(x)` is the predictive mean, :math:`\sigma(x)` is the predictive standard deviation,
    :math:`f^+` is the value of the best observed sample, :math:`\xi` is a small positive "jitter" term to encourage more exploration,
    and :math:`\Phi` is the cumulative distribution function (CDF) of the standard normal distribution.

    In the case of HMC, the function leverages multiple predictive posteriors, each associated
    with a different HMC sample of the GP model parameters, to capture both prediction uncertainty
    and hyperparameter uncertainty. In this setup, the uncertainty in parameters of probabilistic
    mean function (if any) also contributes to the acquisition function values.

    Args:
        rng_key: JAX random number generator key
        model: trained model
        X: new inputs
        best_f:
            Best function value observed so far. Derived from the predictive mean
            when not provided by a user.
        xi: coefficient affecting exploration-exploitation trade-off
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

    moments = _compute_mean_and_var(rng_key, model, X, n, noiseless, **kwargs)

    acq = poi(moments, best_f, xi, maximize)

    if penalty:
        acq -= _compute_penalties(X, recent_points, penalty, penalty_factor, grid_indices)

    return acq


def UE(rng_key: jnp.ndarray, model: Type[ExactGP],
        X: jnp.ndarray,
        n: int = 1,
        noiseless: bool = False,
        penalty: Optional[str] = None,
        recent_points: jnp.ndarray = None,
        grid_indices: jnp.ndarray = None,
        penalty_factor: float = 1.0,
        **kwargs) -> jnp.ndarray:

    r"""
    Uncertainty-based exploration

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Uncertainty-based Exploration (UE) at an input point :math:`x` targets regions where the model's predictions are most uncertain.
    It quantifies this uncertainty as:

    .. math::

        UE(x) = \sigma^2(x)

    where :math:`\sigma^2(x)` is the predictive variance of the model at the input point :math:`x`.

    In the case of HMC, the function leverages multiple predictive posteriors, each associated
    with a different HMC sample of the GP model parameters, to capture both prediction uncertainty
    and hyperparameter uncertainty. In this setup, the uncertainty in parameters of probabilistic
    mean function (if any) also contributes to the acquisition function values.

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

    moments = _compute_mean_and_var(rng_key, model, X, n, noiseless, **kwargs)

    acq = ue(moments)

    if penalty:
        X_ = grid_indices if grid_indices is not None else X
        penalties = compute_penalty(X_, recent_points, penalty, penalty_factor)

        acq -= penalties
    return acq


def KG(rng_key: jnp.ndarray,
       model: Type[ExactGP],
       X: jnp.ndarray,
       n: int = 1,
       maximize: bool = False,
       noiseless: bool = False,
       penalty: Optional[str] = None,
       recent_points: jnp.ndarray = None,
       grid_indices: jnp.ndarray = None,
       penalty_factor: float = 1.0,
       **kwargs) -> jnp.ndarray:
    r"""
    Knowledge gradient

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Knowledge Gradient (KG) at an input point :math:`x` quantifies the expected improvement
    in the optimal decision after observing the function value at :math:`x`.

    The KG value is defined as:

    .. math::

        KG(x) = \mathbb{E}[V_{n+1}^* - V_n^* | x]

    where :math:`V_{n+1}^*` is the optimal expected value of the objective function after \(n+1\) observations and
    :math:`V_n^*` is the optimal expected value of the objective function based on the current \(n\) observations.

    Args:
        rng_key:
            JAX random number generator key for sampling simulated observations
        model:
            Trained model
        X:
            New inputs
        n:
            Number of simulated samples for each point in X
        maximize:
            If True, assumes that BO is solving maximization problem
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
    samples = model.get_samples()

    if model.mcmc is None:
        acq = kg(model, X, samples, rng_key, n, maximize, noiseless, **kwargs)
    else:
        vec_kg = vmap(kg, in_axes=(None, None, 0, 0, None, None, None))
        samples = model.get_samples()
        keys = jra.split(rng_key, num=len(next(iter(samples.values()))))
        acq = vec_kg(model, X, samples, keys, n, maximize, noiseless, **kwargs)

    if penalty:
        acq -= _compute_penalties(X, recent_points, penalty, penalty_factor, grid_indices)

    return acq


def Thompson(rng_key: jnp.ndarray,
             model: Type[ExactGP],
             X: jnp.ndarray, n: int = 1,
             noiseless: bool = False,
             **kwargs) -> jnp.ndarray:
    """
    Thompson sampling.

    For MAP approximation, it draws a single sample of a function from the
    posterior predictive distribution. In the case of HMC, it draws a single posterior
    sample from the HMC samples of GP model parameters and then samples a function from it.

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
