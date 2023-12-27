"""
optimize.py
==============

Optimize continuous acquisition functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

import jax.numpy as jnp
import jax.random as jra


def optimize_acq(rng_key, model, acq_fn, num_initial_guesses, lower_bound, upper_bound, **kwargs):

    try:
        import jaxopt  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "You need to install `jaxopt` to be able to use this feature. "
            "It can be installed with `pip install jaxopt`."
        ) from e

    def acq(x):
        obj = -acq_fn(rng_key, model, jnp.array([x])[None], **kwargs)
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
        if isinstance(x, (list, tuple, float)):
            x = jnp.array([x]) if isinstance(x, float) else jnp.array(x)
        else:
            raise TypeError(f"Expected input to be a list, tuple, float, or jnp.ndarray, got {type(x)} instead.")
    return x
