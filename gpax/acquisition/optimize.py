"""
optimize.py
==============

Optimize continuous acquisition functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from typing import Type, Callable, Union, List, Tuple

import jax.numpy as jnp
import jax.random as jra
import numpy as onp

from ..models.gp import ExactGP


def optimize_acq(rng_key: jnp.ndarray,
                 model: Type[ExactGP],
                 acq_fn: Callable,
                 num_initial_guesses: int,
                 lower_bound: Union[List, Tuple, float, onp.ndarray, jnp.ndarray],
                 upper_bound: Union[List, Tuple, float, onp.ndarray, jnp.ndarray],
                 **kwargs) -> jnp.ndarray:
    """
    Optimizes an acquisition function for a given Gaussian Process model using the JAXopt library.

    This function finds the point that maximizes the acquisition function within the specified bounds.
    It uses L-BFGS-B algorithm through ScipyBoundedMinimize from JAXopt.

    Args:
        rng_key: A JAX random key for stochastic processes.
        model: The Gaussian Process model to be used.
        acq_fn: The acquisition function to be maximized.
        num_initial_guesses: Number of random initial guesses for the optimization.
        lower_bound: Lower bounds for the optimization.
        upper_bound: Upper bounds for the optimization.
        **kwargs: Additional keyword arguments to be passed to the acquisition function.

    Returns:
        Parameter(s) that maximize the acquisition function within the specified bounds.

    Note:
        Ensure JAXopt is installed to use this function (`pip install jaxopt`).
        The acquisition function is minimized using its negative value to find the maximum.

    Examples:

        Optimize EI given a trained GP model for 1D problem

        >>> acq_fn = gpax.acquisition.EI
        >>> num_initial_guesses = 10
        >>> lower_bound = -2.0
        >>> upper_bound = 2.0
        >>> x_next = gpax.acquisition.optimize_acq(
        >>>    rng_key, gp_model, acq_fn,
        >>>    num_initial_guesses, lower_bound, upper_bound,
        >>>    maximize=False, noiseless=True)
    """

    try:
        import jaxopt  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "You need to install `jaxopt` to be able to use this feature. "
            "It can be installed with `pip install jaxopt`."
        ) from e

    def acq(x):
        x = jnp.array([x])
        x = x[None] if x.ndim == 0 else x
        obj = -acq_fn(rng_key, model, x, **kwargs)
        return jnp.reshape(obj, ())

    lower_bound = ensure_array(lower_bound)
    upper_bound = ensure_array(upper_bound)

    initial_guesses = jra.uniform(
        rng_key, shape=(num_initial_guesses, lower_bound.shape[0]),
        minval=lower_bound, maxval=upper_bound)
    initial_acq_vals = acq_fn(rng_key, model, initial_guesses, **kwargs)
    best_initial_guess = initial_guesses[initial_acq_vals.argmax()].squeeze()

    minimizer = jaxopt.ScipyBoundedMinimize(fun=acq, method='l-bfgs-b')
    result = minimizer.run(best_initial_guess, bounds=(lower_bound, upper_bound))

    return result.params


def ensure_array(x):
    if not isinstance(x, jnp.ndarray):
        if isinstance(x, (list, tuple, float, onp.ndarray)):
            x = jnp.array([x]) if isinstance(x, float) else jnp.array(x)
        else:
            raise TypeError(f"Expected input to be a list, tuple, float, or jnp.ndarray, got {type(x)} instead.")
    return x
