"""
dmfgp.py
========

Gaussian process with deep mean function

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from typing import Callable, Dict, Optional, Tuple, Union, Type

import jax
import jax.numpy as jnp
import jax.random as jra
import numpyro
import numpyro.distributions as dist
import haiku as hk

from . import ExactGP
from .nets import DeterministicNN
from ..utils import put_on_device

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray]

clear_cache = jax._src.dispatch.xla_primitive_callable.cache_clear


class DMFGP(ExactGP):
    """
    Deep mean function Gaussian process class

    Args:
        input_dim:
            Number of input dimensions
        kernel:
            Kernel function ('RBF', 'Matern', 'Periodic', or custom function)
        kernel_prior:
            Optional custom priors over kernel hyperparameters. Use it when passing your custom kernel.
        nn:
            Optional custom neural network.
            Assumes haiku transformed module (e.g. hk.transform(lambda x: HaikuMLP(*args)(x))).
        nn_params:
            Trained parameters of the provided network.
        noise_prior_dist:
            Optional custom prior distribution over the observational noise variance.
            Defaults to LogNormal(0,1).
        lengthscale_prior_dist:
            Optional custom prior distribution over kernel lengthscale.
            Defaults to LogNormal(0, 1).
        jitter:
            Small jitter for the numerical stability. Default: 1e-6
    """

    def __init__(
        self,
        input_dim: int,
        kernel: Union[str, kernel_fn_type],
        nn: Type[hk.Module],
        nn_params: Dict = None,
        kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        noise_prior_dist: Optional[dist.Distribution] = None,
        lengthscale_prior_dist: Optional[dist.Distribution] = None,
        jitter: float = 1e-6
    ) -> None:
        super(DMFGP, self).__init__(input_dim, kernel, None, kernel_prior,
                                    None, None, noise_prior_dist,
                                    lengthscale_prior_dist, jitter)

        self.nn = nn
        self.nn_params = nn_params

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """GP model with partially stochastic NN as its mean function"""
        # Sample kernel parameters
        k_params = self._sample_kernel_params() if self.kernel_prior is None else self.kernel_prior()
        # Sample observational noise variance
        noise = self._sample_noise()

        # Get inputs through a deterministic NN part to compute prior mean function
        f_loc = self.nn.apply(self.nn_params, jra.PRNGKey(0), X)

        # Compute kernel
        k = self.kernel(X, X, k_params, noise, self.jitter)

        # Sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def fit(self,
            X: jnp.ndarray,
            y: jnp.ndarray,
            num_warmup: int = 2000,
            num_samples: int = 2000,
            num_chains: int = 1,
            chain_method: str = "sequential",
            nn_train_epochs: int = 500,
            lr: float = 0.01,
            progress_bar: bool = True,
            print_summary: bool = True,
            device: str = None,
            rng_key: jnp.array = None
            ) -> None:
        """
        Run Hamiltonian Monter Carlo to infer the GP parameters

        Args:
            X: 2D feature vector
            y: 1D target vector
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: 'sequential', 'parallel' or 'vectorized'
            nn_train_epochs:
                number of training epochs for deterministic NN
                if training parameters are not provided at the initialization stage
            lr: learning rate for deterministic NN
            progress_bar: show progress bar
            print_summary: print summary at the end of sampling
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key: random number generator key
        """
        if not self.nn_params:
            print("Training deterministic NN...")
            X = self.set_data(X)
            X, y = put_on_device(device, X, y)
            detnn = DeterministicNN(self.nn, self.kernel_dim, learning_rate=lr)
            detnn.train(X, y[:, None], nn_train_epochs)
            self.nn_params = detnn.params
        print("Training deep mean function Gaussian process...")
        return super().fit(X, y, num_warmup, num_samples, num_chains, chain_method, progress_bar, print_summary, device, rng_key)

    def compute_gp_posterior(self, X_new: jnp.ndarray,
                             X_train: jnp.ndarray, y_train: jnp.ndarray,
                             params: Dict[str, jnp.ndarray],
                             noiseless: bool = True,
                             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns mean and covariance of multivariate normal
        posterior for a single sample of trained MFGP parameters
        """
        noise = params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        # Apply deep mean function
        y_residual = y_train - self.nn.apply(self.nn_params, jra.PRNGKey(0), X_train).squeeze()
        # compute kernel matrices for train and test data
        k_XX = self.kernel(X_train, X_train, params, noise, self.jitter)
        k_pp = self.kernel(X_new, X_new, params, noise_p, self.jitter)
        k_pX = self.kernel(X_new, X_train, params)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))
        # Apply deep mean function
        mean += self.nn.apply(self.nn_params, jra.PRNGKey(0), X_new).squeeze()
        return mean, cov

    def print_summary(self) -> None:
        samples = self.get_samples(1)
        list_of_keys = ["k_scale", "k_length", "noise"]
        numpyro.diagnostics.print_summary(
            {k: v for (k, v) in samples.items() if k in list_of_keys})
