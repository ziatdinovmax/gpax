"""
priors.py
=========

Utility functions for setting priors

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

import inspect

from typing import Union, Dict, Type, Callable

import numpyro
import jax.numpy as jnp


def place_normal_prior(param_name: str, loc: float = 0.0, scale: float = 1.0):
    """
    Samples a value from a normal distribution with the specified mean (loc) and standard deviation (scale),
    and assigns it to a named random variable in the probabilistic model. Can be useful for defining prior mean functions
    in structured Gaussian processes.
    """
    return numpyro.sample(param_name, normal_dist(loc, scale))


def place_lognormal_prior(param_name: str, loc: float = 0.0, scale: float = 1.0):
    """
    Samples a value from a log-normal distribution with the specified mean (loc) and standard deviation (scale),
    and assigns it to a named random variable in the probabilistic model. Can be useful for defining prior mean functions
    in structured Gaussian processes.
    """
    return numpyro.sample(param_name, lognormal_dist(loc, scale))


def place_halfnormal_prior(param_name: str, scale: float = 1.0):
    """
    Samples a value from a half-normal distribution with the specified standard deviation (scale),
    and assigns it to a named random variable in the probabilistic model. Can be useful for defining prior mean functions
    in structured Gaussian processes.
    """
    return numpyro.sample(param_name, halfnormal_dist(scale))


def place_uniform_prior(param_name: str,
                        low: float = None,
                        high: float = None,
                        X: jnp.ndarray = None):
    """
    Samples a value from a uniform distribution with the specified low and high values,
    and assigns it to a named random variable in the probabilistic model. Can be useful for defining prior mean functions
    in structured Gaussian processes.
    """
    d = uniform_dist(low, high, X)
    return numpyro.sample(param_name, d)


def place_gamma_prior(param_name: str,
                      c: float = None,
                      r: float = None,
                      X: jnp.ndarray = None):
    """
    Samples a value from a uniform distribution with the specified concentration (c) and rate (r) values,
    and assigns it to a named random variable in the probabilistic model. Can be useful for defining prior mean functions
    in structured Gaussian processes.
    """
    d = gamma_dist(c, r, X)
    return numpyro.sample(param_name, d)


def normal_dist(loc: float = None, scale: float = None
                ) -> numpyro.distributions.Distribution:
    """
    Generate a Normal distribution based on provided center (loc) and standard deviation (scale) parameters.
    If neither are provided, uses 0 and 1 by default. It can be used to pass custom priors to GP models.

    Examples:

        Assign custom prior to kernel lengthscale during GP model initialization
    
        >>> model = gpax.ExactGP(input_dim, kernel, lengthscale_prior_dist=gpax.priors.normal_dist(5, 1))
    
        Train as usual
    
        >>> model.fit(rng_key, X, y)

    """
    loc = loc if loc is not None else 0.0
    scale = scale if scale is not None else 1.0
    return numpyro.distributions.Normal(loc, scale)


def lognormal_dist(loc: float = None, scale: float = None) -> numpyro.distributions.Distribution:
    """
    Generate a LogNormal distribution based on provided center (loc) and standard deviation (scale) parameters.
    If neither are provided, uses 0 and 1 by default. It can be used to pass custom priors to GP models.

    Examples:

        Assign custom prior to kernel lengthscale during GP model initialization
    
        >>> model = gpax.ExactGP(input_dim, kernel, lengthscale_prior_dist=gpax.priors.lognormal_dist(0, 0.1))
    
        Train as usual
    
        >>> model.fit(rng_key, X, y)

    """
    loc = loc if loc is not None else 0.0
    scale = scale if scale is not None else 1.0
    return numpyro.distributions.LogNormal(loc, scale)


def halfnormal_dist(scale: float = None) -> numpyro.distributions.Distribution:
    """
    Generate a half-normal distribution based on provided standard deviation (scale).
    If none is provided, uses 1.0 by default. It can be used to pass custom priors to GP models.

    Examples:

        Assign custom prior to noise variance during GP model initialization
    
        >>> model = gpax.ExactGP(input_dim, kernel, noise_prior_dist=gpax.priors.halfnormal_dist(0.1))
    
        Train as usual
    
        >>> model.fit(rng_key, X, y)

    """
    scale = scale if scale is not None else 1.0
    return numpyro.distributions.HalfNormal(scale)


def gamma_dist(c: float = None,
               r: float = None,
               input_vec: jnp.ndarray = None
               ) -> numpyro.distributions.Distribution:
    """
    Generate a Gamma distribution based on provided shape (c) and rate (r) parameters. If the shape (c) is not provided,
    it attempts to infer it using the range of the input vector divided by 2. The rate parameter defaults to 1.0 if not provided.
    It can be used to pass custom priors to GP models.

    Examples:

        Assign custom prior to kernel lengthscale during GP model initialization
    
        >>> model = gpax.ExactGP(input_dm, kernel, lengthscale_prior_dist=gpax.priors.gamma_dist(2, 5))
    
        Train as usual
    
        >>> model.fit(rng_key, X, y)

    """
    if c is None:
        if input_vec is not None:
            c = (input_vec.max() - input_vec.min()) / 2
        else:
            raise ValueError("Provide either c or an input array")
    if r is None:
        r = 1.0
    return numpyro.distributions.Gamma(c, r)


def uniform_dist(low: float = None,
                 high: float = None,
                 input_vec: jnp.ndarray = None
                 ) -> numpyro.distributions.Distribution:
    """
    Generate a Uniform distribution based on provided low and high bounds. If one of the bounds is not provided,
    it attempts to infer the missing bound(s) using the minimum or maximum value from the input vector.
    It can be used to pass custom priors to GP models.

    Examples:

        Assign custom prior to kernel lengthscale during GP model initialization
    
        >>> model = gpax.ExactGP(input_dm, kernel, lengthscale_prior_dist=gpax.priors.uniform_dist(1, 3))
    
        Train as usual
    
        >>> model.fit(rng_key, X, y)
    """
    if (low is None or high is None) and input_vec is None:
        raise ValueError(
            "If 'low' or 'high' is not provided, an input array must be provided.")
    low = low if low is not None else input_vec.min()
    high = high if high is not None else input_vec.max()

    return numpyro.distributions.Uniform(low, high)


def auto_priors(func: Callable, params_begin_with: int, dist_type: str = 'normal', loc: float = 0.0, scale: float = 1.0) -> Callable:
    """
    Generates a function that, when invoked, samples from normal or log-normal distributions
    for each parameter of the given deterministic function, except the first one.

    Args:
        func (Callable): The deterministic function for which to set normal or log-normal priors.
        params_begin_with (int): Parameters to account for start from this number.
        loc (float, optional): Mean of the normal or log-normal distribution. Defaults to 0.0.
        scale (float, optional): Standard deviation of the normal or log-normal distribution. Defaults to 1.0.

    Returns:
        A function that, when invoked, returns a dictionary of sampled values
        from normal or log-normal distributions for each parameter of the original function.
    """
    place_prior = place_lognormal_prior if dist_type == 'lognormal' else place_normal_prior

    # Get the names of the parameters of the function excluding the first one (dependent variable)
    params_names = list(inspect.signature(func).parameters.keys())[params_begin_with:]

    def sample_priors() -> Dict[str, Union[float, Type[Callable]]]:
        # Return a dictionary with normal priors for each parameter
        return {name: place_prior(name, loc, scale) for name in params_names}

    return sample_priors


def auto_normal_priors(func: Callable, loc: float = 0.0, scale: float = 1.0) -> Callable:
    """
    Places normal priors over function parameters.

    Args:
        func (Callable): The deterministic function for which to set normal priors.
        loc (float, optional): Mean of the normal distribution. Defaults to 0.0.
        scale (float, optional): Standard deviation of the normal distribution. Defaults to 1.0.

    Returns:
        A function that, when invoked, returns a dictionary of sampled values
        from normal distributions for each parameter of the original function.
    """
    return auto_priors(func, 1, 'normal', loc, scale)


def auto_lognormal_priors(func: Callable, loc: float = 0.0, scale: float = 1.0) -> Callable:
    """
    Places log-normal priors over function parameters.

    Args:
        func (Callable): The deterministic function for which to set log-normal priors.
        loc (float, optional): Mean of the log-normal distribution. Defaults to 0.0.
        scale (float, optional): Standard deviation of the log-normal distribution. Defaults to 1.0.

    Returns:
        A function that, when invoked, returns a dictionary of sampled values
        from log-normal distributions for each parameter of the original function.
    """
    return auto_priors(func, 1, 'lognormal', loc, scale)


def auto_normal_kernel_priors(kernel_fn: Callable, loc: float = 0.0, scale: float = 1.0) -> Callable:
    """
    Places normal priors over the kernel parameters.

    Args:
        func (Callable): The deterministic kernel function for which to set normal priors.
        loc (float, optional): Mean of the normal distribution. Defaults to 0.0.
        scale (float, optional): Standard deviation of the normal distribution. Defaults to 1.0.

    Returns:
        A function that, when invoked, returns a dictionary of sampled values
        from normal distributions for each parameter of the original kernel function.
    """
    return auto_priors(kernel_fn, 2, 'normal', loc, scale)


def auto_lognormal_kernel_priors(kernel_fn: Callable, loc: float = 0.0, scale: float = 1.0) -> Callable:
    """
    Places log-normal priors over the kernel parameters.

    Args:
        func (Callable): The deterministic kernel function for which to set log-normal priors.
        loc (float, optional): Mean of the log-normal distribution. Defaults to 0.0.
        scale (float, optional): Standard deviation of the log-normal distribution. Defaults to 1.0.

    Returns:
        A function that, when invoked, returns a dictionary of sampled values
        from log-normal distributions for each parameter of the original kernel function.
    """
    return auto_priors(kernel_fn, 2, 'lognormal', loc, scale)
