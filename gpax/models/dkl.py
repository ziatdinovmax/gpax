"""
dkl.py
=======

Fully Bayesian implementation of deep kernel learning

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import jit

from .gp import ExactGP


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
        kernel_prior:
            Optional priors over kernel hyperparameters; uses LogNormal(0,1) by default
        nn:
            Custom neural network ('feature extractor'); uses a 3-layer MLP
            with hyperbolic tangent activations by default
        nn_prior:
            Priors over the weights and biases in 'nn'; uses normal priors by default
        latent_prior:
            Optional prior over the latent space (BNN embedding); uses none by default
        hidden_dim:
            Optional custom MLP architecture. For example [16, 8, 4] corresponds to a 3-layer
            neural network backbone containing 16, 8, and 4 neurons activated by tanh(). The latent
            layer is added autoamtically and doesn't have to be spcified here. Defaults to [64, 32].

        **kwargs:
            Optional custom prior distributions over observational noise (noise_dist_prior)
            and kernel lengthscale (lengthscale_prior_dist)


    Examples:

        DKL with image patches as inputs and a 1-d vector as targets

        >>> # Get random number generator keys for training and prediction
        >>> key1, key2 = gpax.utils.get_keys()
        >>> input data dimensions are (n, height*width*channels)
        >>> data_dim = X.shape[-1]
        >>> # Initialize DKL model with 2 latent dimensions
        >>> dkl = gpax.DKL(data_dim, z_dim=2, kernel='RBF')
        >>> # Train model by parallelizing HMC chains on a single GPU
        >>> dkl.fit(key1, X, y, num_warmup=333, num_samples=333, num_chains=3, chain_method='vectorized')
        >>> # Obtain posterior mean and samples from DKL posterior at new inputs
        >>> # using batches to avoid memory overflow
        >>> y_pred, y_samples = dkl.predict_in_batches(key2, X_new)
    """

    def __init__(self, input_dim: int, z_dim: int = 2, kernel: str = 'RBF',
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 nn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 nn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 latent_prior: Optional[Callable[[jnp.ndarray], Dict[str, jnp.ndarray]]] = None,
                 hidden_dim: Optional[List[int]] = None, **kwargs
                 ) -> None:
        super(DKL, self).__init__(input_dim, kernel, None, kernel_prior, **kwargs)
        hdim = hidden_dim if hidden_dim is not None else [64, 32]
        self.nn = nn if nn else get_mlp(hdim)
        self.nn_prior = nn_prior if nn_prior else get_mlp_prior(input_dim, z_dim, hdim)
        self.kernel_dim = z_dim
        self.latent_prior = latent_prior

    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              **kwargs: float
              ) -> None:
        """DKL probabilistic model"""
        jitter = kwargs.get("jitter", 1e-6)
        task_dim = X.shape[0]
        # BNN part
        nn_params = self.nn_prior(task_dim)
        z = self.nn(X, nn_params)
        if self.latent_prior:  # Sample latent variable
            z = self.latent_prior(z)
        # Sample GP kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params(task_dim)
        # Sample noise
        noise = self._sample_noise()
        # GP's mean function
        f_loc = jnp.zeros(z.shape[0])
        # compute kernel
        k = self.kernel(z, z, kernel_params, noise, jitter=jitter)
        # Sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def get_mvn_posterior(self,
                          X_new: jnp.ndarray,
                          params: Dict[str, jnp.ndarray],
                          noiseless: bool = False,
                          **kwargs: float
                          ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        noise = params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        # embed data into the latent space
        z_train = self.nn(self.X_train, params)
        z_new = self.nn(X_new, params)
        # compute kernel matrices for train and new ('test') data
        k_pp = self.kernel(z_new, z_new, params, noise_p, **kwargs)
        k_pX = self.kernel(z_new, z_train, params, jitter=0.0)
        k_XX = self.kernel(z_train, z_train, params, noise, **kwargs)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, self.y_train))
        return mean, cov

    @partial(jit, static_argnames='self')
    def embed(self, X_new: jnp.ndarray) -> jnp.ndarray:
        """
        Embeds data into the latent space using the inferred weights
        of the DKL's Bayesian neural network
        """
        samples = self.get_samples(chain_dim=False)
        predictive = jax.vmap(lambda params: self.nn(X_new, params))
        z = predictive(samples)
        return z

    def _print_summary(self):
        list_of_keys = ["k_scale", "k_length", "noise", "period"]
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary(
            {k: v for (k, v) in samples.items() if k in list_of_keys})


def sample_weights(name: str, in_channels: int, out_channels: int, task_dim: int) -> jnp.ndarray:
    """Sampling weights matrix"""
    w = numpyro.sample(name=name, fn=dist.Normal(
        loc=jnp.zeros((in_channels, out_channels)),
        scale=jnp.ones((in_channels, out_channels))))
    return w


def sample_biases(name: str, channels: int, task_dim: int) -> jnp.ndarray:
    """Sampling bias vector"""
    b = numpyro.sample(name=name, fn=dist.Normal(
        loc=jnp.zeros((channels)), scale=jnp.ones((channels))))
    return b


def get_mlp(architecture: List[int]) -> Callable:
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


def get_mlp_prior(input_dim: int, output_dim: int, architecture: List[int]) -> Dict[str, jnp.ndarray]:
    """Priors over weights and biases for a Bayesian MLP"""
    def mlp_prior(task_dim: int):
        params = {}
        in_channels = input_dim
        for i, out_channels in enumerate(architecture):
            params[f"w{i}"] = sample_weights(f"w{i}", in_channels, out_channels, task_dim)
            params[f"b{i}"] = sample_biases(f"b{i}", out_channels, task_dim)
            in_channels = out_channels
        # Output layer
        params[f"w{len(architecture)}"] = sample_weights(f"w{len(architecture)}", in_channels, output_dim, task_dim)
        params[f"b{len(architecture)}"] = sample_biases(f"b{len(architecture)}", output_dim, task_dim)
        return params
    return mlp_prior
