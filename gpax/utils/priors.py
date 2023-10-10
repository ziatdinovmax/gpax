"""
priors.py
=========

Utility functions for setting priors

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import inspect
import re

from typing import Union, Dict, Type, List, Callable, Optional

import numpyro
import jax
import jax.numpy as jnp

from ..kernels.kernels import square_scaled_distance, add_jitter, _sqrt


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
    I neithere are provided, uses 0 and 1 by default.
    """
    loc = loc if loc is not None else 0.0
    scale = scale if scale is not None else 1.0
    return numpyro.distributions.Normal(loc, scale)


def lognormal_dist(loc: float = None, scale: float = None) -> numpyro.distributions.Distribution:
    """
    Generate a LogNormal distribution based on provided center (loc) and standard deviation (scale) parameters.
    I neithere are provided, uses 0 and 1 by default.
    """
    loc = loc if loc is not None else 0.0
    scale = scale if scale is not None else 1.0
    return numpyro.distributions.LogNormal(loc, scale)


def halfnormal_dist(scale: float = None) -> numpyro.distributions.Distribution:
    """
    Generate a half-normal distribution based on provided standard deviation (scale).
    If none is provided, uses 1.0 by default.
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
    """
    if (low is None or high is None) and input_vec is None:
        raise ValueError(
            "If 'low' or 'high' is not provided, an input array must be provided.")
    low = low if low is not None else input_vec.min()
    high = high if high is not None else input_vec.max()

    return numpyro.distributions.Uniform(low, high)


def set_fn(func: Callable) -> Callable:
    """
    Transforms the given deterministic function to use a params dictionary
    for its parameters, excluding the first one (assumed to be the dependent variable).

    Args:
    - func (Callable): The deterministic function to be transformed.

    Returns:
    - Callable: The transformed function where parameters are accessed
                from a `params` dictionary.
    """
    # Extract parameter names excluding the first one (assumed to be the dependent variable)
    params_names = list(inspect.signature(func).parameters.keys())[1:]

    # Create the transformed function definition
    transformed_code = f"def {func.__name__}(x, params):\n"

    # Retrieve the source code of the function and indent it to be a valid function body
    source = inspect.getsource(func).split("\n", 1)[1]
    source = "    " + source.replace("\n", "\n    ")

    # Replace each parameter name with its dictionary lookup using regex
    for name in params_names:
        source = re.sub(rf'\b{name}\b', f'params["{name}"]', source)

    # Combine to get the full source
    transformed_code += source

    # Define the transformed function in the local namespace
    local_namespace = {}
    exec(transformed_code, globals(), local_namespace)

    # Return the transformed function
    return local_namespace[func.__name__]


def set_kernel_fn(func: Callable,
                  independent_vars: List[str] = ["X", "Z"],
                  jit_decorator: bool = True,
                  docstring: Optional[str] = None) -> Callable:
    """
    Transforms the given kernel function to use a params dictionary for its hyperparameters.
    The resultant function will always add jitter before returning the computed kernel.

    Args:
        func (Callable): The kernel function to be transformed.
        independent_vars (List[str], optional): List of independent variable names in the function. Defaults to ["X", "Z"].
        jit_decorator (bool, optional): @jax.jit decorator to be applied to the transformed function. Defaults to True.
        docstring (Optional[str], optional): Docstring to be added to the transformed function. Defaults to None.

    Returns:
        Callable: The transformed kernel function where hyperparameters are accessed from a `params` dictionary.
    """

    # Extract parameter names excluding the independent variables
    params_names = [k for k, v in inspect.signature(func).parameters.items() if v.default == v.empty]
    for var in independent_vars:
        params_names.remove(var)

    transformed_code = ""
    if jit_decorator:
        transformed_code += "@jit" + "\n"

    additional_args = "noise: int = 0, jitter: float = 1e-6, **kwargs"
    transformed_code += f"def {func.__name__}({', '.join(independent_vars)}, params: Dict[str, jnp.ndarray], {additional_args}):\n"

    if docstring:
        transformed_code += '    """' + docstring + '"""\n'

    source = inspect.getsource(func).split("\n", 1)[1]
    lines = source.split("\n")

    for idx, line in enumerate(lines):
        # Convert all parameter names to their dictionary lookup throughout the function body
        for name in params_names:
            lines[idx] = re.sub(rf'\b{name}\b', f'params["{name}"]', lines[idx])

    # Combine lines back and then split again by return
    modified_source = '\n'.join(lines)
    pre_return, return_statement = modified_source.split('return', 1)

    # Append custom jitter code
    custom_code = f"    {pre_return.strip()}\n    k = {return_statement.strip()}\n"
    custom_code += """
    if X.shape == Z.shape:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k
    """

    transformed_code += custom_code

    local_namespace = {"jit": jax.jit}
    exec(transformed_code, globals(), local_namespace)

    return local_namespace[func.__name__]


def _set_noise_kernel_fn(func: Callable) -> Callable:
    """
    Modifies the GPax kernel function to append "_noise" after "k" in dictionary keys it accesses.

    Args:
        func (Callable): Original function.

    Returns:
        Callable: Modified function.
    """

    # Get the source code of the function
    source = inspect.getsource(func)

    # Split the source into decorators, definition, and body
    decorators_and_def, body = source.split("\n", 1)

    # Replace all occurrences of params["k with params["k_noise in the body
    modified_body = re.sub(r'params\["k', 'params["k_noise', body)

    # Combine decorators, definition, and modified body
    modified_source = f"{decorators_and_def}\n{modified_body}"

    # Define local namespace including the jit decorator
    local_namespace = {"jit": jax.jit}

    # Execute the modified source to redefine the function in the provided namespace
    exec(modified_source, globals(), local_namespace)

    # Return the modified function
    return local_namespace[func.__name__]


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
