"""
kernels.py
==========

Kernel functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Union, Dict, Callable

import math

import jax.numpy as jnp
from jax import jit, vmap

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],  jnp.ndarray]


def _sqrt(x, eps=1e-12):
    return jnp.sqrt(x + eps)


def add_jitter(x, jitter=1e-6):
    return x + jitter


def square_scaled_distance(X: jnp.ndarray, Z: jnp.ndarray,
                           lengthscale: Union[jnp.ndarray, float] = 1.
                           ) -> jnp.ndarray:
    r"""
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
              noise: int = 0, jitter: float = 1e-6,
              **kwargs) -> jnp.ndarray:
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
        k += add_jitter(noise, jitter) * jnp.eye(X.shape[0])
    return k


@jit
def MaternKernel(X: jnp.ndarray, Z: jnp.ndarray,
                 params: Dict[str, jnp.ndarray],
                 noise: int = 0, jitter: float = 1e-6,
                 **kwargs) -> jnp.ndarray:
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
        k += add_jitter(noise, jitter) * jnp.eye(X.shape[0])
    return k


@jit
def PeriodicKernel(X: jnp.ndarray, Z: jnp.ndarray,
                   params: Dict[str, jnp.ndarray],
                   noise: int = 0, jitter: float = 1e-6,
                   **kwargs
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
        k += add_jitter(noise, jitter) * jnp.eye(X.shape[0])
    return k


def nngp_erf(x1: jnp.ndarray, x2: jnp.ndarray,
             var_b: jnp.array, var_w: jnp.array,
             depth: int = 3) -> jnp.array:
    """
    Computes the Neural Network Gaussian Process (NNGP) kernel value for
    a single pair of inputs using the Erf activation.

    Args:
        x1: First input vector.
        x2: Second input vector.
        var_b: Bias variance.
        var_w: Weight variance.
        depth: The number of layers in the corresponding infinite-width neural network.
               Controls the level of recursion in the computation.

    Returns:
        Kernel value for the pair of inputs.
    """
    d = x1.shape[-1]
    if depth == 0:
        return var_b + var_w * jnp.sum(x1 * x2, axis=-1) / d
    else:
        K_12 = nngp_erf(x1, x2, var_b, var_w, depth - 1)
        K_11 = nngp_erf(x1, x1, var_b, var_w, depth - 1)
        K_22 = nngp_erf(x2, x2, var_b, var_w, depth - 1)
        sqrt_term = jnp.sqrt((1 + 2 * K_11) * (1 + 2 * K_22))
        fraction = 2 * K_12 / sqrt_term
        epsilon = 1e-7
        theta = jnp.arcsin(jnp.clip(fraction, a_min=-1 + epsilon, a_max=1 - epsilon))
        result = var_b + 2 * var_w / jnp.pi * theta
        return result


def nngp_relu(x1: jnp.ndarray, x2: jnp.ndarray,
              var_b: jnp.array, var_w: jnp.array,
              depth: int = 3) -> jnp.array:
    """
    Computes the Neural Network Gaussian Process (NNGP) kernel value for
    a single pair of inputs using RELU activation.

    Args:
        x1: First input vector.
        x2: Second input vector.
        var_b: Bias variance.
        var_w: Weight variance.
        depth: The number of layers in the corresponding infinite-width neural network.
               Controls the level of recursion in the computation.

    Returns:
        Kernel value for the pair of inputs.
    """
    eps = 1e-7
    d = x1.shape[-1]
    if depth == 0:
        return var_b + var_w * jnp.sum(x1 * x2, axis=-1) / d
    else:
        K_12 = nngp_relu(x1, x2, var_b, var_w, depth - 1, )
        K_11 = nngp_relu(x1, x1, var_b, var_w, depth - 1, )
        K_22 = nngp_relu(x2, x2, var_b, var_w, depth - 1, )
        sqrt_term = jnp.sqrt(K_11 * K_22)
        fraction = K_12 / sqrt_term
        theta = jnp.arccos(jnp.clip(fraction, a_min=-1 + eps, a_max=1 - eps))
        theta_term = jnp.sin(theta) + (jnp.pi - theta) * fraction
        return var_b + var_w / (2 * jnp.pi) * sqrt_term * theta_term


def NNGPKernel(activation: str = 'erf', depth: int = 3
               ) -> Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]:
    """
    Neural Network Gaussian Process (NNGP) kernel function

    Args:
        activation: activation function ('erf' or 'relu')
        depth: The number of layers in the corresponding infinite-width neural network.
               Controls the level of recursion in the computation.

    Returns:
        Function for computing kernel matrix between X and Z.
    """
    nngp_single_pair_ = nngp_relu if activation == 'relu' else nngp_erf

    def NNGPKernel_func(X: jnp.ndarray, Z: jnp.ndarray,
                        params: Dict[str, jnp.ndarray],
                        noise: jnp.ndarray = 0, jitter: float = 1e-6,
                        **kwargs
                        ) -> jnp.ndarray:
        """
        Computes the Neural Network Gaussian Process (NNGP) kernel.

        Args:
            X: First set of input vectors.
            Z: Second set of input vectors.
            params: Dictionary containing bias variance and weight variance

        Returns:
            Computed kernel matrix between X and Z.
        """
        var_b = params["var_b"]
        var_w = params["var_w"]
        k = vmap(lambda x: vmap(lambda z: nngp_single_pair_(x, z, var_b, var_w, depth))(Z))(X)
        if X.shape == Z.shape:
            k += add_jitter(noise, jitter) * jnp.eye(X.shape[0])
        return k

    return NNGPKernel_func


def get_kernel(kernel: Union[str, kernel_fn_type] = 'RBF', **kwargs):
    kernel_book = {
        'RBF': RBFKernel,
        'Matern': MaternKernel,
        'Periodic': PeriodicKernel,
        'NNGP': NNGPKernel(**kwargs)
    }
    if isinstance(kernel, str):
        try:
            kernel = kernel_book[kernel]
        except KeyError:
            print('Select one of the currently available kernels:',
                  *kernel_book.keys())
            raise
    return kernel
