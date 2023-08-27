"""
acquisition.py
==============

Acquisition functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Type, Optional, Callable, Any

import jax.numpy as jnp
import jax.random as jra
from jax import vmap
import numpy as onp

from ..models.gp import ExactGP
from .base_acq import ei, ucb, poi, ue, kg
from .penalties import compute_penalty


def compute_acquisition(
        model: Type[ExactGP],
        X: jnp.ndarray,
        acq_func: Callable[..., jnp.ndarray],
        *acq_args: Any,
        penalty: Optional[str] = None,
        recent_points: Optional[jnp.ndarray] = None,
        grid_indices: Optional[jnp.ndarray] = None,
        penalty_factor: float = 1.0,
        **kwargs) -> jnp.ndarray:
    """
    Computes acquistion function of a given type

    Args:
        model: The trained model.
        X: New inputs.
        acq_func: Acquisition function to be used (e.g., ei or ucb).
        *acq_args: Positional arguments passed to the acquisition function.
        penalty:
            Penalty applied to the acquisition function to discourage re-evaluation
            at or near points that were recently evaluated.
        recent_points:
            An array of recently visited points [oldest, ..., newest] provided by user
        grid_indices:
            Grid indices of data points in X array for the penalty term calculation.
            For example, if each data point is an image patch, the indices could correspond
            to the (i, j) pixel coordinates of their centers in the original image.
        penalty_factor:
            Penalty factor :math:`\lambda` in :math:`\alpha - \lambda \cdot \pi(X, r)`
        **kwargs:
            Additional keyword arguments passed to the acquisition function.

    Returns:
        Computed acquisition function values
    """
    if penalty and not isinstance(recent_points, (onp.ndarray, jnp.ndarray)):
        raise ValueError("Please provide an array of recently visited points")

    X = X[:, None] if X.ndim < 2 else X
    samples = model.get_samples()

    if model.mcmc is None:
        acq = acq_func(model, X, samples, *acq_args, **kwargs)
    else:
        f = vmap(acq_func, in_axes=(None, None, 0) + (None,)*len(acq_args))
        acq = f(model, X, samples, *acq_args, **kwargs)
        acq = acq.mean(0)

    if penalty:
        X_ = grid_indices if grid_indices is not None else X
        penalties = compute_penalty(X_, recent_points, penalty, penalty_factor)
        acq -= penalties

    return acq


def EI(rng_key: jnp.ndarray, model: Type[ExactGP],
       X: jnp.ndarray,
       maximize: bool = False,
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

    where:
    - :math:`\mu(x)` is the predictive mean.
    - :math:`\sigma(x)` is the predictive standard deviation.
    - :math:`f^+` is the value of the best observed sample.
    - :math:`Z` is defined as:

    .. math::

        Z = \frac{\mu(x) - f^+}{\sigma(x)}

    provided :math:`\sigma(x) > 0`.


    Args:
        rng_key: JAX random number generator key
        model: trained model
        X: new inputs
        maximize: If True, assumes that BO is solving maximization problem
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
    if rng_key is not None:
        import warnings
        warnings.warn("`rng_key` is deprecated and will be removed in future versions. "
                      "It's no longer used.", DeprecationWarning, stacklevel=2)
    return compute_acquisition(
        model, X, ei, maximize, noiseless,
        penalty=penalty, recent_points=recent_points,
        grid_indices=grid_indices, penalty_factor=penalty_factor,
        **kwargs)


def POI(rng_key: jnp.ndarray,
        model: Type[ExactGP],
        X: jnp.ndarray,
        xi: float = 0.01,
        maximize: bool = False,
        noiseless: bool = False,
        penalty: Optional[str] = None,
        recent_points: jnp.ndarray = None,
        grid_indices: jnp.ndarray = None,
        penalty_factor: float = 1.0,
        **kwargs) -> jnp.ndarray:
    r"""
    Probability of Improvement

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Probability of Improvement (PI) at an input point :math:`x` is defined as:

    .. math::

        PI(x) = \Phi\left(\frac{\mu(x) - f^+ - \xi}{\sigma(x)}\right)

    where:
    - :math:`\mu(x)` is the predictive mean.
    - :math:`\sigma(x)` is the predictive standard deviation.
    - :math:`f^+` is the value of the best observed sample.
    - :math:`\xi` is a small positive "jitter" term to encourage more exploration.
    - :math:`\Phi` is the cumulative distribution function (CDF) of the standard normal distribution.

    Args:
        rng_key: JAX random number generator key
        model: trained model
        X: new inputs
        xi: exploration-exploitation tradeoff (defaults to 0.01)
        maximize: If True, assumes that BO is solving maximization problem
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
    if rng_key is not None:
        import warnings
        warnings.warn("`rng_key` is deprecated and will be removed in future versions. "
                      "It's no longer used.", DeprecationWarning, stacklevel=2)
    return compute_acquisition(
        model, X, poi, xi, maximize, noiseless,
        penalty=penalty, recent_points=recent_points,
        grid_indices=grid_indices, penalty_factor=penalty_factor,
        **kwargs)


def UCB(rng_key: jnp.ndarray,
        model: Type[ExactGP],
        X: jnp.ndarray,
        beta: float = .25,
        maximize: bool = False,
        noiseless: bool = False,
        penalty: Optional[str] = None,
        recent_points: jnp.ndarray = None,
        grid_indices: jnp.ndarray = None,
        penalty_factor: float = 1.0,
        **kwargs) -> jnp.ndarray:
    r"""
    Upper confidence bound

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Upper Confidence Bound (UCB) at an input point :math:`x` is defined as:

    .. math::

        UCB(x) = \mu(x) + \kappa \sigma(x)

    where:
    - :math:`\mu(x)` is the predictive mean.
    - :math:`\sigma(x)` is the predictive standard deviation.
    - :math:`\kappa` is the exploration-exploitation trade-off parameter.

    Args:
        rng_key: JAX random number generator key
        model: trained model
        X: new inputs
        beta: coefficient balancing exploration-exploitation trade-off
        maximize: If True, assumes that BO is solving maximization problem
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
    if rng_key is not None:
        import warnings
        warnings.warn("`rng_key` is deprecated and will be removed in future versions. "
                      "It's no longer used.", DeprecationWarning, stacklevel=2)
    return compute_acquisition(
        model, X, ucb, beta, maximize, noiseless,
        penalty=penalty, recent_points=recent_points,
        grid_indices=grid_indices, penalty_factor=penalty_factor,
        **kwargs)


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

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Uncertainty-based Exploration (UE) at an input point :math:`x` targets regions where the model's predictions are most uncertain.
    It quantifies this uncertainty as:

    .. math::

        UE(x) = \sigma^2(x)

    where:
    - :math:`\sigma^2(x)` is the predictive variance of the model at the input point :math:`x`.


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
    if rng_key is not None:
        import warnings
        warnings.warn("`rng_key` is deprecated and will be removed in future versions. "
                      "It's no longer used.", DeprecationWarning, stacklevel=2)
    return compute_acquisition(
        model, X, ue, noiseless,
        penalty=penalty, recent_points=recent_points,
        grid_indices=grid_indices, penalty_factor=penalty_factor,
        **kwargs)


def KG(model: Type[ExactGP],
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
    the Knowledge Gradient (KG) at an input point :math:`x` quantifies the expected improvement in the optimal decision after observing the function value at :math:`x`.

    The KG value is defined as:

    .. math::

        KG(x) = \mathbb{E}[V_{n+1}^* - V_n^* | x]

    where:
    - :math:`V_{n+1}^*` is the optimal expected value of the objective function after \(n+1\) observations.
    - :math:`V_n^*` is the optimal expected value of the objective function based on the current \(n\) observations.

    Args:
        model: trained model
        X: new inputs
        n: number of simulated samples for each point in X
        maximize: If True, assumes that BO is solving maximization problem
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
        **rng_key: JAX random number generator key for sampling simulated observations
        **jitter:
            Small positive term added to the diagonal part of a covariance
            matrix for numerical stability (Default: 1e-6)
    """
    return compute_acquisition(
        model, X, kg, n, maximize, noiseless,
        penalty=penalty, recent_points=recent_points,
        grid_indices=grid_indices, penalty_factor=penalty_factor,
        **kwargs)


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

