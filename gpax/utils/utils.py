"""
utils.py
========

Utility functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Union, Dict, Type, List

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
