"""
utils.py
========

Utility functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import inspect
import re
from typing import Union, Dict, Type, List, Callable

import jax
import jax.numpy as jnp
import numpy as onp

import numpyro


def enable_x64():
    """Use double (x64) precision for jax arrays"""
    jax.config.update("jax_enable_x64", True)


def get_keys(seed: int = 0):
    """
    Simple wrapper for jax.random.split to get
    rng keys for model inference and prediction
    """
    rng_key_1, rng_key_2 = jax.random.split(jax.random.PRNGKey(seed))
    return rng_key_1, rng_key_2


def split_in_batches(X_new: Union[onp.ndarray, jnp.ndarray],
                     batch_size: int = 100, dim: int = 0):
    """
    Splits array into batches along the first or second dimensions
    """
    if dim not in [0, 1]:
        raise NotImplementedError("'dim' must be equal to 0 or 1")
    num_batches = jax.numpy.floor_divide(X_new.shape[dim], batch_size)
    X_split = []
    for i in range(num_batches):
        if dim == 0:
            X_i = X_new[i*batch_size:(i+1)*batch_size]
        else:
            X_i = X_new[:, i*batch_size:(i+1)*batch_size]
        X_split.append(X_i)
    X_i = X_new[(i+1)*batch_size:] if dim == 0 else X_new[:, (i+1)*batch_size:]
    if X_i.shape[dim] > 0:
        X_split.append(X_i)
    return X_split


def split_dict(data: Dict[str, jnp.ndarray], chunk_size: int
               ) -> List[Dict[str, jnp.ndarray]]:
    """Splits a dictionary of arrays into a list of smaller dictionaries.

    Args:
        data: Dictionary containing numpy arrays.
        chunk_size: Desired size of the smaller arrays.

    Returns:
        List of dictionaries with smaller numpy arrays.
    """

    # Get the length of the arrays
    N = len(next(iter(data.values())))

    # Calculate number of chunks
    num_chunks = int(onp.ceil(N / chunk_size))

    # Split the dictionary
    result = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i+1) * chunk_size, N)

        chunk = {key: value[start_idx:end_idx] for key, value in data.items()}
        result.append(chunk)

    return result


def random_sample_dict(data: Dict[str, jnp.ndarray],
                       num_samples: int,
                       rng_key: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Returns a dictionary with a smaller number of consistent random samples for each array.

    Args:
        data: Dictionary containing numpy arrays.
        num_samples: Number of random samples required.
        rng_key: Random number generator key

    Returns:
        Dictionary with the consistently sampled arrays.
    """

    # Generate unique random indices
    num_data_points = len(next(iter(data.values())))
    indices = jax.random.permutation(rng_key, num_data_points)[:num_samples]

    return {key: value[indices] for key, value in data.items()}


def get_haiku_dict(kernel_params: Dict[str, jnp.ndarray]) -> Dict[str, Dict[str, jnp.ndarray]]:
    """
    Extracts weights and biases from viDKL dictionary into a separate
    dictionary compatible with haiku's .apply() method
    """
    all_weights = {}
    all_biases = {}
    for key, val in kernel_params.items():
        if key.startswith('feature_extractor'):
            name_split = key.split('/')
            name_new = name_split[1] + '/' + name_split[2][:-2]
            if name_split[2][-1] == 'b':
                all_biases[name_new] = val
            else:
                all_weights[name_new] = val
    nn_params = {}
    for (k, v1), (_, v2) in zip(all_weights.items(), all_biases.items()):
        nn_params[k] = {"w": v1, "b": v2}
    return nn_params


def dviz(d: Type[numpyro.distributions.Distribution], samples: int = 1000) -> None:
    """
    Utility function for visualizing numpyro distributions

    Args:
        d: numpyro distribution; e.g. numpyro.distributions.Gamma(2, 2)
        samples: number of samples
    """
    try:
        import seaborn as sns  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "You need to install `seaborn` to be able to use this feature. "
            "It can be installed with `pip install seaborn`."
        ) from e
    import matplotlib.pyplot as plt

    with numpyro.handlers.seed(rng_seed=0):
        samples = d.sample(jax.random.PRNGKey(0), sample_shape=(samples,))
    plt.figure(dpi=100)
    sns.histplot(samples, kde=True, fill=False)
    plt.show()


def preprocess_sparse_image(sparse_image):
    """
    Creates GP inputs from sparse image data where missing values are represented by zeros.
    If your actual data contains zeros, you will need to (re-)normalize it.
    Otherwise, those elements will be interpreted as missng values. The function returns
    two arrays of the shapes (N, D) and (N,) that are used as training inputs and targets in GP
    and an array of full indices of the shape (N_full, D) for reconstructing the full image. D is
    the image dimensionality (D=2 for a 2D image)
    """
    # Find non-zero element indices
    non_zero_indices = onp.nonzero(sparse_image)
    # Create the GP input using the indices
    gp_input = onp.column_stack(non_zero_indices)
    # Extract non-zero values (targets) from the sparse image
    targets = sparse_image[non_zero_indices]
    # Generate indices for the entire image
    full_indices = onp.array(onp.meshgrid(*[onp.arange(dim) for dim in sparse_image.shape])).T.reshape(-1, sparse_image.ndim)
    return gp_input, targets, full_indices


def place_normal_prior(param_name: str, loc: float = 0.0, scale: float = 1.0):
    """
    Samples a value from a normal distribution with the specified mean (loc) and standard deviation (scale),
    and assigns it to a named random variable in the probabilistic model. Can be useful for defining prior mean functions
    in structured Gaussian processes.
    """
    return numpyro.sample(param_name, normal_dist(loc, scale))


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
    

def auto_normal_priors(func: Callable, loc: float = 0.0, scale: float = 1.0) -> Callable:
    """
    Generates a function that, when invoked, samples from normal distributions
    for each parameter of the given deterministic function, except the first one.

    Args:
    - func (Callable): The deterministic function for which to set normal priors.
    - loc (float, optional): Mean of the normal distribution. Defaults to 0.0.
    - scale (float, optional): Standard deviation of the normal distribution. Defaults to 1.0.

    Returns:
    - Callable: A function that, when invoked, returns a dictionary of sampled values
                from normal distributions for each parameter of the original function.
    """
    # Get the names of the parameters of the function excluding the first one (dependent variable)
    params_names = list(inspect.signature(func).parameters.keys())[1:]

    def sample_priors() -> Dict[str, Union[float, Type[Callable]]]:
        # Return a dictionary with normal priors for each parameter
        return {name: place_normal_prior(name, loc, scale) for name in params_names}

    return sample_priors
