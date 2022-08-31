"""
dkl.py
=======

Fully Bayesian implementation of deep kernel learning

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import jit

from .vgp import vExactGP
from .kernels import get_kernel


class DKL(vExactGP):
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
                 latent_prior: Optional[Callable[[jnp.ndarray], Dict[str, jnp.ndarray]]] = None
                 ) -> None:
        super(DKL, self).__init__(input_dim, kernel, kernel_prior)
        self.nn = nn if nn else mlp
        self.nn_prior = nn_prior if nn_prior else mlp_prior(input_dim, z_dim)
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
        bnn_params = self.nn_prior(task_dim)
        z = jax.jit(jax.vmap(self.nn))(X, bnn_params)
        if self.latent_prior:  # Sample latent variable
            z = self.latent_prior(z)
        # Sample GP kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params(task_dim)
        # Sample noise
        with numpyro.plate('obs_noise', task_dim):
            noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
        # GP's mean function
        f_loc = jnp.zeros(z.shape[:2])
        # compute kernel(s)
        jitter = jnp.array(jitter).repeat(task_dim)
        k_args = (z, z, kernel_params, noise)
        k = jax.vmap(get_kernel(self.kernel))(*k_args, jitter=jitter)
        # Sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    @partial(jit, static_argnames='self')
    def _get_mvn_posterior(self,
                           X_train: jnp.ndarray, y_train: jnp.ndarray,
                           X_new: jnp.ndarray, params: Dict[str, jnp.ndarray],
                           noiseless: bool = False, **kwargs: float
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        noise = params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        # embed data into the latent space
        z_train = self.nn(X_train, params)
        z_new = self.nn(X_new, params)
        # compute kernel matrices for train and new ('test') data
        k_pp = get_kernel(self.kernel)(z_new, z_new, params, noise_p, **kwargs)
        k_pX = get_kernel(self.kernel)(z_new, z_train, params, jitter=0.0)
        k_XX = get_kernel(self.kernel)(z_train, z_train, params, noise, **kwargs)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_train))
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

    def _set_data(self,
                  X: jnp.ndarray,
                  y: Optional[jnp.ndarray] = None
                  ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X[None] if X.ndim == 2 else X  # add task pseudo-dimension
        if y is not None:
            y = y[None] if y.ndim == 1 else y  # add task pseudo-dimension
            return X, y
        return X

    def _print_summary(self):
        list_of_keys = ["k_scale", "k_length", "noise", "period"]
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary(
            {k: v for (k, v) in samples.items() if k in list_of_keys})


def sample_weights(name: str, in_channels: int, out_channels: int, task_dim: int) -> jnp.ndarray:
    """Sampling weights matrix"""
    with numpyro.plate("batch_dim", task_dim, dim=-3):
        w = numpyro.sample(name=name, fn=dist.Normal(
            loc=jnp.zeros((in_channels, out_channels)),
            scale=jnp.ones((in_channels, out_channels))))
    return w


def sample_biases(name: str, channels: int, task_dim: int) -> jnp.ndarray:
    """Sampling bias vector"""
    with numpyro.plate("batch_dim", task_dim, dim=-3):
        b = numpyro.sample(name=name, fn=dist.Normal(
            loc=jnp.zeros((channels)), scale=jnp.ones((channels))))
    return b


def mlp(X: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Simple MLP for a single MCMC sample of weights and biases"""
    h1 = jnp.tanh(jnp.matmul(X, params["w1"]) + params["b1"])
    h2 = jnp.tanh(jnp.matmul(h1, params["w2"]) + params["b2"])
    z = jnp.matmul(h2, params["w3"]) + params["b3"]
    return z


def mlp_prior(input_dim: int, zdim: int = 2) -> Dict[str, jnp.array]:
    """Priors over weights and biases in the default Bayesian MLP"""
    hdim = [64, 32]

    def _bnn_prior(task_dim: int):
        w1 = sample_weights("w1", input_dim, hdim[0], task_dim)
        b1 = sample_biases("b1", hdim[0], task_dim)
        w2 = sample_weights("w2", hdim[0], hdim[1], task_dim)
        b2 = sample_biases("b2", hdim[1], task_dim)
        w3 = sample_weights("w3", hdim[1], zdim, task_dim)
        b3 = sample_biases("b3", zdim, task_dim)
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}

    return _bnn_prior
