"""
vidkl.py
========

Variational inference-based implementation of deep kernel learning

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from typing import Optional, Type, List

import jax.numpy as jnp
import jax.random as jra
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import haiku as hk

from .dkl import DKL
from ..utils import get_haiku_compatible_dict, put_on_device


class viDKL(DKL):
    """
    Variational Inference-based Deep Kernel Learning

    Args:
        input_dim:
            Number of input dimensions
        z_dim:
            Latent space dimensionality (defaults to 2)
        kernel:
            Kernel function ('RBF', 'Matern', 'Periodic', or custom function)
        nn:
            Optionally provide a custom neural network in haiku; if not provided,
            the model uses a 3-layer MLP with hyperbolic tangent activations by default
        hidden_dim:
            Optional custom MLP architecture. For example [16, 8, 4] corresponds to a 3-layer
            neural network backbone containing 16, 8, and 4 neurons activated by tanh(). The latent
            layer is added autoamtically and doesn't have to be specified here. Defaults to [32, 16, 8].

        **kwargs:
            Optional custom prior distributions over observational noise (noise_dist_prior)
            and kernel lengthscale (lengthscale_prior_dist) or over the entire kernel if custom kernel is used.

        Examples:

        vi-DKL with image patches as inputs and a 1-d vector as targets

        >>> # Since we use MLP by default, the input data dimensions are (n, height*width*channels)
        >>> data_dim = X.shape[-1]
        >>> # Initialize vi-DKL model with 2 latent dimensions
        >>> dkl = gpax.viDKL(input_dim=data_dim, z_dim=2, kernel='RBF')
        >>> Train a model
        >>> dkl.fit(X_train, y_train, num_steps=1000, step_size=0.005)
        >>> # Obtain posterior mean and variance ('uncertainty') at new inputs
        >>> y_mean, y_var = dkl.predict(X_new)
    """

    def __init__(self, input_dim: int, z_dim: int = 2, kernel: str = 'RBF',
                 hidden_dim: Optional[List[int]] = None, activation: str = 'relu',
                 nn: Type[hk.Module] = None, **kwargs
                 ) -> None:
        super(viDKL, self).__init__(input_dim, z_dim, kernel, hidden_dim, activation, nn, **kwargs)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_steps: int = 1000, step_size: float = 5e-3,
            progress_bar: bool = True,
            print_summary: bool = True,
            device: str = None,
            rng_key: jnp.array = None,
            **kwargs: float
            ) -> None:
        """
        Run variational inference to learn DKL (hyper)parameters

        Args:
            rng_key: random number generator key
            X: 2D feature vector with *(number of points, number of features)* dimensions
            y: 1D target vector with *(n,)* dimensions
            num_steps: number of SVI steps
            step_size: step size schedule for Adam optimizer
            progress_bar: show progress bar
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            print_summary: print summary at the end of training
            rng_key: random number generator key
        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X, y = self.set_data(X, y)
        X, y = put_on_device(device, X, y)
        self.X_train = X
        self.y_train = y

        optim = numpyro.optim.Adam(step_size=step_size, b1=0.5)
        self.svi = SVI(
            self.model,
            guide=AutoDelta(self.model),
            optim=optim,
            loss=Trace_ELBO(),
            X=X,
            y=y,
            **kwargs
        )

        params = self.svi.run(
            key, num_steps, progress_bar=progress_bar)[0]

        self.params = self.svi.guide.median(params)

        if print_summary:
            self.print_summary()

    def get_samples(self, **kwargs):
        return get_haiku_compatible_dict(self.params, map=True)  # map=True adds an extra batch dimension to make it work with vmap

    def print_summary(self, print_nn_weights: bool = False) -> None:
        list_of_keys = ["k_scale", "k_length", "noise"]
        print('\nInferred GP kernel parameters')
        for (k, vals) in self.params.items():
            if k in list_of_keys:
                spaces = " " * (15 - len(k))
                print(k, spaces, jnp.around(vals, 4))