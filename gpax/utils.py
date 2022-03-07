from typing import Union, Dict

import jax
import jax.numpy as jnp
import numpy as onp

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
