"""
vi_ibnn.py
=======

Infinite width Bayesian neural net (variational approximation)

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Optional, Dict, Callable

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .vigp import viGP
from ..kernels import get_kernel


class vi_iBNN(viGP):
    """
    Variational inference-based infinite Bayesian neural net (vi-iBNN)

    Args:
        input_dim:
            Number of input dimensions
        depth:
            The number of layers in the corresponding infinite-width neural network.
        activation:
            activation function ('erf' or 'relu')
        mean_fn:
            Optional deterministic mean function (use 'mean_fn_priors' to make it probabilistic)
        nngp_prior:
            Optional custom priors over NNGP kernel hyperparameters.
            The defaults are HalfNormal(1) for bias variance and LogNormal(0, 10) for weight variance.
        mean_fn_prior:
            Optional priors over mean function parameters
        noise_prior:
            Optional custom prior for the observation noise variance;
            uses LogNormal(0,1) by default.
    """

    def __init__(self, input_dim: int, depth: int = 3, activation: str = 'erf',
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 nngp_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None
                 ) -> None:
        args = (input_dim, None, mean_fn, nngp_prior, mean_fn_prior, noise_prior)
        super(vi_iBNN, self).__init__(*args)
        self.kernel = get_kernel("NNGP", activation=activation, depth=depth)

    def _sample_kernel_params(self) -> Dict[str, jnp.ndarray]:
        """
        Sample NNGP kernel parameters with default
        weakly-informative log-normal priors
        """
        var_b = numpyro.sample("var_b", dist.HalfNormal(1))
        var_w = numpyro.sample("var_w", dist.LogNormal(0, 10))
        return {"var_b": var_b, "var_w": var_w}
