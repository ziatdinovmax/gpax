from typing import Union

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

