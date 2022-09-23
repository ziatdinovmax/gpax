"""
kernels.py
==========

Kernel functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Union, Dict, Callable

import math

import jax.numpy as jnp
from jax import jit

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],  jnp.ndarray]


def _sqrt(x, eps=1e-12):
    return jnp.sqrt(x + eps)


def add_jitter(x, jitter=1e-6):
    return x + jitter


def square_scaled_distance(X: jnp.ndarray, Z: jnp.ndarray,
                           lengthscale: Union[jnp.ndarray, float] = 1.
                           ) -> jnp.ndarray:
    """
    Computes a square of scaled distance, :math:`\|\frac{X-Z}{l}\|^2`,
    between X and Z are vectors with :math:`n x num_features` dimensions
    """
    scaled_X = X / lengthscale
    scaled_Z = Z / lengthscale
    X2 = (scaled_X ** 2).sum(1, keepdims=True)
    Z2 = (scaled_Z ** 2).sum(1, keepdims=True)
    XZ = jnp.matmul(scaled_X, scaled_Z.T)
    r2 = X2 - 2 * XZ + Z2.T
    return r2.clip(0)


@jit
def RBFKernel(X: jnp.ndarray, Z: jnp.ndarray,
              params: Dict[str, jnp.ndarray],
              noise: int = 0, **kwargs: float) -> jnp.ndarray:
    """
    Radial basis function kernel

    Args:
        X: 2D vector with *(number of points, number of features)* dimension
        Z: 2D vector with *(number of points, number of features)* dimension
        params: Dictionary with kernel hyperparameters 'k_length' and 'k_scale'
        noise: optional noise vector with dimension (n,)

    Returns:
        Computed kernel matrix betwenen X and Z
    """
    r2 = square_scaled_distance(X, Z, params["k_length"])
    k = params["k_scale"] * jnp.exp(-0.5 * r2)
    if X.shape == Z.shape:
        k += add_jitter(noise, **kwargs) * jnp.eye(X.shape[0])
    return k


@jit
def MaternKernel(X: jnp.ndarray, Z: jnp.ndarray,
                 params: Dict[str, jnp.ndarray],
                 noise: int = 0, **kwargs: float) -> jnp.ndarray:
    """
    Matern52 kernel

    Args:
        X: 2D vector with *(number of points, number of features)* dimension
        Z: 2D vector with *(number of points, number of features)* dimension
        params: Dictionary with kernel hyperparameters 'k_length' and 'k_scale'
        noise: optional noise vector with dimension (n,)

    Returns:
        Computed kernel matrix between X and Z
    """
    r2 = square_scaled_distance(X, Z, params["k_length"])
    r = _sqrt(r2)
    sqrt5_r = 5**0.5 * r
    k = params["k_scale"] * (1 + sqrt5_r + (5/3) * r2) * jnp.exp(-sqrt5_r)
    if X.shape == Z.shape:
        k += add_jitter(noise, **kwargs) * jnp.eye(X.shape[0])
    return k


@jit
def PeriodicKernel(X: jnp.ndarray, Z: jnp.ndarray,
                   params: Dict[str, jnp.ndarray],
                   noise: int = 0, **kwargs: float
                   ) -> jnp.ndarray:
    """
    Periodic kernel

    Args:
        X: 2D vector with *(number of points, number of features)* dimension
        Z: 2D vector with *(number of points, number of features)* dimension
        params: Dictionary with kernel hyperparameters 'k_length', 'k_scale', and 'period'
        noise: optional noise vector with dimension (n,)

    Returns:
        Computed kernel matrix between X and Z
    """
    d = X[:, None] - Z[None]
    scaled_sin = jnp.sin(math.pi * d / params["period"]) / params["k_length"]
    k = params["k_scale"] * jnp.exp(-2 * (scaled_sin ** 2).sum(-1))
    if X.shape == Z.shape:
        k += add_jitter(noise, **kwargs) * jnp.eye(X.shape[0])
    return k


def get_kernel(kernel: Union[str, kernel_fn_type] = 'RBF'):
    kernel_book = {
        'RBF': RBFKernel,
        'Matern': MaternKernel,
        'Periodic': PeriodicKernel
    }
    if isinstance(kernel, str):
        try:
            kernel = kernel_book[kernel]
        except KeyError:
            print('Select one of the currently available kernels:',
                  *kernel_book.keys())
            raise
    return kernel
