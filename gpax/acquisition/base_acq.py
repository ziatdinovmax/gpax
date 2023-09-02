"""
base_acq.py
==============

Base acquisition functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Type, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from ..models.gp import ExactGP
from ..utils import get_keys


def ei(moments: Tuple[jnp.ndarray, jnp.ndarray],
       best_f: float = None,
       maximize: bool = False,
       **kwargs) -> jnp.ndarray:
    r"""
    Expected Improvement

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Expected Improvement at an input point :math:`x` is defined as:

    .. math::
        EI(x) =
        \begin{cases}
        (\mu(x) - f^+ - \xi) \Phi(Z) + \sigma(x) \phi(Z) & \text{if } \sigma(x) > 0 \\
        0 & \text{if } \sigma(x) = 0
        \end{cases}

    where:
    - :math:`\mu(x)` is the predictive mean.
    - :math:`\sigma(x)` is the predictive standard deviation.
    - :math:`f^+` is the value of the best observed sample.
    - :math:`\xi` is a small positive "jitter" term (not used in this function).
    - :math:`Z` is defined as:

    .. math::

        Z = \frac{\mu(x) - f^+ - \xi}{\sigma(x)}

    provided :math:`\sigma(x) > 0`.

    Args:
        moments:
            Tuple with predictive mean and variance
            (first and second moments of predictive distribution).
        best_f:
            Best function value observed so far. Derived from the predictive mean
            when not provided by a user.
        maximize:
            If True, assumes that BO is solving maximization problem.
    """
    mean, var = moments
    if best_f is None:
        best_f = mean.max() if maximize else mean.min()
    sigma = jnp.sqrt(var)
    u = (mean - best_f) / sigma
    if not maximize:
        u = -u
    normal = dist.Normal(jnp.zeros_like(u), jnp.ones_like(u))
    ucdf = normal.cdf(u)
    updf = jnp.exp(normal.log_prob(u))
    acq = sigma * (updf + u * ucdf)
    return acq


def ucb(moments: Tuple[jnp.ndarray, jnp.ndarray],
        beta: float = 0.25,
        maximize: bool = False,
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
        moments:
            Tuple with predictive mean and variance
            (first and second moments of predictive distribution).
        maximize: If True, assumes that BO is solving maximization problem
        beta: coefficient balancing exploration-exploitation trade-off
    """
    mean, var = moments
    delta = jnp.sqrt(beta * var)
    if maximize:
        acq = mean + delta
    else:
        acq = -(mean - delta)  # return a negative acq for argmax in BO
    return acq


def ue(moments: Tuple[jnp.ndarray, jnp.ndarray], **kwargs) -> jnp.ndarray:
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
        moments:
            Tuple with predictive mean and variance
            (first and second moments of predictive distribution).

    """
    _, var = moments
    return jnp.sqrt(var)


def poi(moments: Tuple[jnp.ndarray, jnp.ndarray],
        best_f: float = None, xi: float = 0.01,
        maximize: bool = False, **kwargs) -> jnp.ndarray:
    r"""
    Probability of Improvement

    Args:
        moments:
            Tuple with predictive mean and variance
            (first and second moments of predictive distribution).
        maximize: If True, assumes that BO is solving maximization problem
        xi: Exploration-exploitation trade-off parameter (Defaults to 0.01)
    """
    mean, var = moments
    if best_f is None:
        best_f = mean.max() if maximize else mean.min()
    sigma = jnp.sqrt(var)
    u = (mean - best_f - xi) / sigma
    if not maximize:
        u = -u
    normal = dist.Normal(jnp.zeros_like(u), jnp.ones_like(u))
    return normal.cdf(u)


def kg(model: Type[ExactGP],
       X_new: jnp.ndarray,
       sample: Dict[str, jnp.ndarray],
       rng_key: Optional[jnp.ndarray] = None,
       n: int = 10,
       maximize: bool = True,
       noiseless: bool = True,
       **kwargs):
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
