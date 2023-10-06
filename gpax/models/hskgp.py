"""
hskgp.py
=========

Fully Bayesian implementation of heteroskedastic Gaussian process regression

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from typing import Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from . import ExactGP
from ..kernels import get_kernel

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray]

clear_cache = jax._src.dispatch.xla_primitive_callable.cache_clear


class VarNoiseGP(ExactGP):
    """
    Heteroskedastic Gaussian process class

    Args:
        input_dim:
            Number of input dimensions
        kernel:
            Main kernel function ('RBF', 'Matern', 'Periodic', or custom function)
        noise_kernel:
            Noise kernel function ('RBF', 'Matern', 'Periodic', or custom function)
        mean_fn:
            Optional deterministic mean function (use 'mean_fn_priors' to make it probabilistic)
        kernel_prior:
            Optional custom priors over main kernel hyperparameters. Use it when passing your custom kernel.
        mean_fn_prior:
            Optional priors over mean function parameters
        lengthscale_prior_dist:
            Optional custom prior distribution over main kernel lengthscale. Defaults to LogNormal(0, 1).
        noise_lengthscale_prior_dist:
            Optional custom prior distribution over noise kernel lengthscale. Defaults to LogNormal(0, 1).
    """

    def __init__(
        self,
        input_dim: int,
        kernel: Union[str, kernel_fn_type],
        noise_kernel: Union[str, kernel_fn_type] = 'RBF',
        mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
        kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        lengthscale_prior_dist: Optional[dist.Distribution] = None,
        noise_lengthscale_prior_dist: Optional[dist.Distribution] = None
    ) -> None:
        args = (input_dim, kernel, mean_fn, kernel_prior, mean_fn_prior, None, None, lengthscale_prior_dist)
        super(VarNoiseGP, self).__init__(*args)
        self.noise_kernel = get_kernel(noise_kernel)
        self.noise_lengthscale_prior_dist = noise_lengthscale_prior_dist

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs: float) -> None:
        """GP probabilistic model with inputs X and targets y"""
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[0])

        # Sample noise kernel parameters
        noise_kernel_params = self._sample_noise_kernel_params()
        # Compute noise kernel
        k_noise = self.noise_kernel(X, X, noise_kernel_params, 0, **kwargs)
        # Compute log variance of the data points
        points_log_var = numpyro.sample(
            "log_var",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k_noise)
        )

        # Sample main kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params()
        # Add mean function (if any)
        if self.mean_fn is not None:
            args = [X]
            if self.mean_fn_prior is not None:
                args += [self.mean_fn_prior()]
            f_loc += self.mean_fn(*args).squeeze()
        # compute main kernel
        k = self.kernel(X, X, kernel_params, 0, **kwargs)
        # Sample y according to the standard Gaussian process formula. Note that instead of adding a fixed noise term to the kernel,
        # we exponentiate the log_var samples to get the variance at each data point
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k+jnp.diag(jnp.exp(points_log_var))),
            obs=y,
        )

    def _sample_noise_kernel_params(self) -> Dict[str, jnp.ndarray]:
        """
        Sample noise kernel parameters
        """
        if self.noise_lengthscale_prior_dist is not None:
            noise_length_dist = self.noise_lengthscale_prior_dist
        else:
            noise_length_dist = dist.LogNormal(0, 1)
        noise_scale = numpyro.sample("k_noise_scale", dist.LogNormal(0, 1))
        noise_length = numpyro.sample("k_noise_length", noise_length_dist)
        return {"k_length": noise_length, "k_scale": noise_scale}

    def get_mvn_posterior(
        self, X_new: jnp.ndarray, params: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns parameters (mean and cov) of multivariate normal posterior
        for a single sample of GP parameters
        """
        # Main GP part
        y_residual = self.y_train.copy()
        if self.mean_fn is not None:
            args = [self.X_train, params] if self.mean_fn_prior else [self.X_train]
            y_residual -= self.mean_fn(*args).squeeze()
        # compute main kernel matrices for train and test data
        k_pp = self.kernel(X_new, X_new, params, 0, **kwargs)
        k_pX = self.kernel(X_new, self.X_train, params, jitter=0.0)
        k_XX = self.kernel(self.X_train, self.X_train, params, 0, **kwargs)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))
        if self.mean_fn is not None:
            args = [X_new, params] if self.mean_fn_prior else [X_new]
            mean += self.mean_fn(*args).squeeze()

        # Noise GP part
        # Compute noise kernel matrices
        k_pX_noise = self.noise_kernel(
            X_new, self.X_train,
            {"k_length": params["k_noise_length"], "k_scale": params["k_noise_scale"]},
            jitter=0.0)
        k_XX_noise = self.noise_kernel(
            self.X_train, self.X_train,
            {"k_length": params["k_noise_length"], "k_scale": params["k_noise_scale"]},
            0, **kwargs)
        # Compute noise predictive mean
        K_xx_noise_inv = jnp.linalg.inv(k_XX_noise)
        predicted_log_var = jnp.matmul(k_pX_noise, jnp.matmul(K_xx_noise_inv, params["log_var"]))
        predicted_noise_variance = jnp.exp(predicted_log_var)

        # Return the main GP's predictive mean and combined (main + noise) covariance matrix
        return mean, cov + jnp.diag(predicted_noise_variance)

    def get_data_var_samples(self):
        """Returns inferred (training) data variance samples"""
        samples = self.mcmc.get_samples()
        return jnp.exp(samples["log_var"])

    def _print_summary(self):
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary({k: v for (k, v) in samples.items() if 'log_var' not in k})
