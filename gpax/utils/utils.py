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


def split_in_batches(array: jnp.ndarray, batch_size: int = 200) -> List[jnp.ndarray]:
    """Splits array into batches"""
    num_batches = (array.shape[0] + batch_size - 1) // batch_size
    return [array[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


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
    dtype = sparse_image.dtype
    # Find non-zero element indices
    non_zero_indices = onp.nonzero(sparse_image)
    # Create the GP input using the indices
    gp_input = onp.column_stack(non_zero_indices)
    # Extract non-zero values (targets) from the sparse image
    targets = sparse_image[non_zero_indices]
    # Generate indices for the entire image
    full_indices = onp.array(onp.meshgrid(*[onp.arange(dim) for dim in sparse_image.shape])).T.reshape(-1, sparse_image.ndim)
    return gp_input.astype(dtype), targets.astype(dtype), full_indices.astype(dtype)


def initialize_inducing_points(X, ratio=0.1, method='uniform', key=None):
    """
    Initialize inducing points for a sparse Gaussian Process in JAX.

    Parameters:
    - X: A (n_samples, num_features) array of training data.
    - ratio: A float between 0 and 1 indicating the fraction of inducing points.
    - method: A string indicating the method for selecting inducing points ('uniform', 'random', 'kmeans').
    - key: A JAX random key, required if method is 'random'.

    Returns:
    - inducing_points: A subset of X used as inducing points.
    """
    if not 0 < ratio < 1:
        raise ValueError("The 'ratio' value must be between 0 and 1")

    n_samples = X.shape[0]
    n_inducing = int(n_samples * ratio)

    if method == 'uniform':
        indices = jnp.linspace(0, n_samples - 1, n_inducing, dtype=jnp.int8)
        inducing_points = X[indices]
    elif method == 'random':
        if key is None:
            raise ValueError("A JAX random key must be provided for random selection")
        indices = jax.random.choice(key, n_samples, shape=(n_inducing,), replace=False)
        inducing_points = X[indices]
    elif method == 'kmeans':
        try:
            from sklearn.cluster import KMeans  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "You need to install `seaborn` to be able to use this feature. "
                "It can be installed with `pip install scikit-learn`."
            ) from e
        # Use sklearn for KMeans clustering, then convert result to JAX array
        kmeans = KMeans(n_clusters=n_inducing, random_state=0).fit(X)
        inducing_points = jnp.array(kmeans.cluster_centers_)
    else:
        raise ValueError("Method must be 'uniform', 'random', or 'kmeans'")

    return inducing_points


def infer_device(device_preference: str = None):
    """
    Returns a JAX device based on the specified preference.
    Defaults to the first available device if no preference is given, or if the specified
    device type is not available.

    Args:
    - device_preference (str, optional): The preferred device type ('cpu' or 'gpu').

    Returns:
    - A JAX device.
    """
    if device_preference:
        # Normalize the input to lowercase to ensure compatibility.
        device_preference = device_preference.lower()
        # Try to get devices of the specified type.
        devices_of_type = jax.devices(device_preference)
        if devices_of_type:
            # If there are any devices of the requested type, return the first one.
            return devices_of_type[0]
        else:
            print(f"No devices of type '{device_preference}' found. Falling back to the default device.")

    # If no preference is specified or no devices of the specified type are found, return the default device.
    return jax.devices()[0]


def put_on_device(device=None, *data_items):
    """
    Places multiple data items on the specified device.

    Args:
        device: The target device as a string (e.g., 'cpu', 'gpu'). If None, the default device is used.
        *data_items: Variable number of data items (such as JAX array or dictionary) to be placed on the device.

    Returns:
        A tuple of the data items placed on the specified device. The structure of each data item is preserved.
    """
    if device is not None:
        device = infer_device(device)
        return tuple(jax.device_put(item, device) for item in data_items)
    return data_items
