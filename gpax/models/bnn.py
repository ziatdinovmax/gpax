"""
bnn.py
=======

Fully Bayesian MLPs

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from typing import Callable, Dict, Optional, List, Union, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .spm import sPM


class BNN(sPM):
    """Fully Bayesian MLP"""
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 noise_prior_dist: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 hidden_dim: Optional[List[int]] = None, **kwargs):
        hidden_dim = [64, 32] if not hidden_dim else hidden_dim
        nn = kwargs.get("nn", get_mlp(hidden_dim))
        nn_prior = kwargs.get("nn_prior", get_mlp_prior(input_dim, output_dim, hidden_dim))
        super(BNN, self).__init__(nn, nn_prior, None, noise_prior_dist)

    def _set_data(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None
                  ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            y = y[:, None] if y.ndim < 2 else y
            return X, y
        return X


def sample_weights(name: str, in_channels: int, out_channels: int) -> jnp.ndarray:
    """Sampling weights matrix"""
    w = numpyro.sample(name=name, fn=dist.Normal(
        loc=jnp.zeros((in_channels, out_channels)),
        scale=jnp.ones((in_channels, out_channels))))
    return w


def sample_biases(name: str, channels: int) -> jnp.ndarray:
    """Sampling bias vector"""
    b = numpyro.sample(name=name, fn=dist.Normal(
        loc=jnp.zeros((channels)), scale=jnp.ones((channels))))
    return b


def get_mlp(architecture: List[int]) -> Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]:
    """Returns a function that represents an MLP for a given architecture."""
    def mlp(X: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """MLP for a single MCMC sample of weights and biases, handling arbitrary number of layers."""
        h = X
        for i in range(len(architecture)):
            h = jnp.tanh(jnp.matmul(h, params[f"w{i}"]) + params[f"b{i}"])
        # No non-linearity after the last layer
        z = jnp.matmul(h, params[f"w{len(architecture)}"]) + params[f"b{len(architecture)}"]
        return z
    return mlp


def get_mlp_prior(input_dim: int, output_dim: int, architecture: List[int]) -> Callable[[], Dict[str, jnp.ndarray]]:
    """Priors over weights and biases for a Bayesian MLP"""
    def mlp_prior():
        params = {}
        in_channels = input_dim
        for i, out_channels in enumerate(architecture):
            params[f"w{i}"] = sample_weights(f"w{i}", in_channels, out_channels)
            params[f"b{i}"] = sample_biases(f"b{i}", out_channels)
            in_channels = out_channels
        # Output layer
        params[f"w{len(architecture)}"] = sample_weights(f"w{len(architecture)}", in_channels, output_dim)
        params[f"b{len(architecture)}"] = sample_biases(f"b{len(architecture)}", output_dim)
        return params
    return mlp_prior
