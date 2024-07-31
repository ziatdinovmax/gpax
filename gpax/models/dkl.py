"""
dkl.py
=======

Fully Bayesian implementation of deep kernel learning

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union, List, Type

import jax
import jax.numpy as jnp
import jax.random as jra
from jax import vmap
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_haiku_module
from jax import jit
import haiku as hk

from .gp import ExactGP
from .nets import HaikuMLP
from ..utils import get_haiku_compatible_dict


class DKL(ExactGP):
    """
    Fully Bayesian implementation of deep kernel learning

    Args:
        input_dim:
            Number of input dimensions
        z_dim:
            Latent space dimensionality (defaults to 2)
        kernel:
            Kernel function ('RBF', 'Matern', 'Periodic', or custom function)
        hidden_dim:
            Optional custom MLP architecture. For example [16, 8, 4] corresponds to a 3-layer
            neural network backbone containing 16, 8, and 4 neurons activated by tanh(). The latent
            layer is added autoamtically and doesn't have to be specified here. Defaults to [32, 16, 8].
        activation:
            Nonlinear activation function for NN. Defaults to 'relu'.
        nn:
            Optionally provide a custom neural network in haiku; if not provided,
            the model uses a 3-layer MLP with hyperbolic tangent activations by default

        **kwargs:
            Optional custom prior distributions over observational noise (noise_dist_prior)
            and kernel lengthscale (lengthscale_prior_dist) or over the entire kernel if custom kernel is used.


    Examples:

        DKL with image patches as inputs and a 1-d vector as targets

        >>> input data dimensions are (n, height*width*channels)
        >>> data_dim = X.shape[-1]
        >>> # Initialize DKL model with 2 latent dimensions
        >>> dkl = gpax.DKL(data_dim, z_dim=2, kernel='RBF')
        >>> # Train model by parallelizing HMC chains on a single GPU
        >>> dkl.fit(X, y, num_warmup=333, num_samples=333, num_chains=3, chain_method='vectorized')
        >>> # Obtain posterior mean and samples from DKL posterior at new inputs
        >>> # using batches to avoid memory overflow
        >>> y_pred, y_samples = dkl.predict_in_batches(key2, X_new)
    """

    def __init__(self, input_dim: int, z_dim: int = 2, kernel: str = 'RBF',
                 hidden_dim: Optional[List[int]] = None, activation: str = 'tanh',
                 nn: Type[hk.Module] = None, **kwargs
                 ) -> None:
        super(DKL, self).__init__(z_dim, kernel, **kwargs)
        if nn is not None:
            self.nn_module = hk.transform(lambda x: nn()(x))
        else:
            hdim = hidden_dim if hidden_dim is not None else [32, 16, 8]
            self.nn_module = hk.transform(lambda x: HaikuMLP(hdim, z_dim, activation)(x))
        self.data_dim = (input_dim,) if isinstance(input_dim, int) else input_dim

    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              **kwargs: float
              ) -> None:
        """DKL probabilistic model"""
        jitter = kwargs.get("jitter", 1e-6)
        # BNN part
        feature_extractor = random_haiku_module(
                "feature_extractor", self.nn_module, input_shape=(1, *self.data_dim),
                prior=(lambda name, shape: dist.Cauchy() if name.startswith("b") else dist.Normal()))
        z = feature_extractor(X)
        # Sample GP kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params()
        # Sample noise
        noise = self._sample_noise()
        # GP's mean function
        f_loc = jnp.zeros(z.shape[0])
        # Compute kernel
        k = self.kernel(z, z, kernel_params, noise, jitter=jitter)
        # Sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        samples = self.mcmc.get_samples(group_by_chain=chain_dim)
        # Get NN weights and biases compatible with haiku prediction methods
        return get_haiku_compatible_dict(samples)

    def compute_gp_posterior(self,
                             X_new: jnp.ndarray,
                             X_train: jnp.ndarray,
                             y_train: jnp.ndarray,
                             params: Dict[str, jnp.ndarray],
                             noiseless: bool = True
                             ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # Transform X_new and X_train using the neural network to get embeddings
        z_new = self.nn_module.apply(params, jra.PRNGKey(0), X_new)
        z_train = self.nn_module.apply(params, jra.PRNGKey(0), X_train)

        # Proceed with the original GP computations using the embedded inputs
        return super().compute_gp_posterior(z_new, z_train, y_train, params, noiseless)

    def embed(self, X_new: jnp.ndarray) -> jnp.ndarray:
        """
        Embeds data into the latent space using the inferred weights
        of the DKL's Bayesian neural network
        """
        X_new = self.set_data(X_new)
        samples = self.get_samples()
        predictive = vmap(lambda params: self.nn_module.apply(params, jra.PRNGKey(0), X_new))
        z = predictive(samples)
        return z

    def print_summary(self) -> None:
        samples = self.get_samples(1)
        list_of_keys = ["k_scale", "k_length", "noise"]
        numpyro.diagnostics.print_summary(
            {k: v for (k, v) in samples.items() if k in list_of_keys})
