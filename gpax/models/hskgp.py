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
from ..utils import _set_noise_kernel_fn

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray]


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
        noise_kernel_prior:
            Optional custom priors over noise kernel hyperparameters. Use it when passing your custom kernel.
        lengthscale_prior_dist:
            Optional custom prior distribution over main kernel lengthscale. Defaults to LogNormal(0, 1).
        noise_mean_fn:
            Optional noise mean function
        noise_mean_fn_prior:
            Optional priors over noise mean function
        noise_lengthscale_prior_dist:
            Optional custom prior distribution over noise kernel lengthscale. Defaults to LogNormal(0, 1).
     Examples:

        Use two different kernels with default priors for main and noise processes

        >>> # Get random number generator keys for training and prediction
        >>> rng_key, rng_key_predict = gpax.utils.get_keys()
        >>> # Initialize model
        >>> gp_model = gpax.VarNoiseGP(input_dim=1, kernel='RBF, noise_kernel='Matern')
        >>> # Run HMC to obtain posterior samples for the GP model parameters
        >>> gp_model.fit(rng_key, X, y)
        >>> # Make a prediction on new inputs
        >>> y_pred, y_samples = gp_model.predict(rng_key_predict, X_new)
        >>> # Get the inferred noise samples (for training data)
        >>> data_variance = gp_model.get_data_var_samples()

        Specify custom kernel lengthscale priors for main and noise kernels

        >>> lscale_prior = gpax.utils.gamma_dist(5, 1)  # equivalent to numpyro.distributions.Gamma(5, 1)
        >>> noise_lscale_prior = gpax.utils.halfnormal_dist(1)  # equivalent to numpyro.distributions.HalfNormal(1)
        >>> # Initialize model
        >>> gp_model = gpax.VarNoiseGP(
        >>>    input_dim=1, kernel='RBF, noise_kernel='Matern',
        >>>    lengthscale_prior_dist=lscale_prior, noise_lengthscale_prior_dist=noise_lscale_prior)
        >>> # Run HMC to obtain posterior samples for the GP model parameters
        >>> gp_model.fit(rng_key, X, y)
        >>> # Make a prediction on new inputs
        >>> y_pred, y_samples = gp_model.predict(rng_key_predict, X_new)
        >>> # Get the inferred noise samples (for training data)
        >>> data_variance = gp_model.get_data_var_samples()
    """
    def __init__(
        self,
        input_dim: int,
        kernel: Union[str, kernel_fn_type],
        noise_kernel: Union[str, kernel_fn_type] = 'RBF',
        mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
        kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        noise_kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        lengthscale_prior_dist: Optional[dist.Distribution] = None,
        noise_mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
        noise_mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        noise_lengthscale_prior_dist: Optional[dist.Distribution] = None
    ) -> None:
        args = (input_dim, kernel, mean_fn, kernel_prior, mean_fn_prior, None, None, lengthscale_prior_dist)
        super(VarNoiseGP, self).__init__(*args)
        noise_kernel_ = get_kernel(noise_kernel)
        self.noise_kernel = _set_noise_kernel_fn(noise_kernel_) if isinstance(noise_kernel, str) else noise_kernel_

        self.noise_mean_fn = noise_mean_fn
        self.noise_mean_fn_prior = noise_mean_fn_prior
        self.noise_kernel_prior = noise_kernel_prior
        self.noise_lengthscale_prior_dist = noise_lengthscale_prior_dist

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs: float) -> None:
        """Heteroskedastic GP probabilistic model with inputs X and targets y"""
        # Initialize mean functions at zeros
        f_loc = jnp.zeros(X.shape[0])
        noise_f_loc = jnp.zeros(X.shape[0])

        # Sample noise kernel parameters
        if self.noise_kernel_prior:
            noise_kernel_params = self.noise_kernel_prior()
        else:
            noise_kernel_params = self._sample_noise_kernel_params()
        # Add noise prior mean function (if any)
        if self.noise_mean_fn is not None:
            args = [X]
            if self.noise_mean_fn_prior is not None:
                args += [self.noise_mean_fn_prior()]
            noise_f_loc += jnp.log(self.noise_mean_fn(*args)).squeeze()
        # Compute noise kernel
        k_noise = self.noise_kernel(X, X, noise_kernel_params, 0, **kwargs)
        # Compute log variance of the data points
        points_log_var = numpyro.sample(
            "log_var",
            dist.MultivariateNormal(loc=noise_f_loc, covariance_matrix=k_noise)
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
        # Compute main kernel
        k = self.kernel(X, X, kernel_params, 0, **kwargs)
        # Sample y according to the standard Gaussian process formula. Note that instead of adding a fixed noise term to the kernel as in regular GP,
        # we exponentiate and take a diagonal of the log_var samples to get the variance at each data point and add that variance to the main kernel
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
        return {"k_noise_length": noise_length, "k_noise_scale": noise_scale}

    def get_mvn_posterior(
        self, X_new: jnp.ndarray, params: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns parameters (mean and cov) of multivariate normal posterior
        for a single sample of heteroskedastic GP parameters
        """
        # Main GP part
        y_residual = self.y_train.copy()
        if self.mean_fn is not None:
            args = [self.X_train, params] if self.mean_fn_prior else [self.X_train]
            y_residual -= self.mean_fn(*args).squeeze()
        # Compute main kernel matrices for train and test data
        k_pp = self.kernel(X_new, X_new, params, 0, **kwargs)
        k_pX = self.kernel(X_new, self.X_train, params, jitter=0.0)
        k_XX = self.kernel(self.X_train, self.X_train, params, 0, **kwargs)
        # Compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))
        if self.mean_fn is not None:
            args = [X_new, params] if self.mean_fn_prior else [X_new]
            mean += self.mean_fn(*args).squeeze()

        # Noise GP part
        # Compute noise kernel matrices
        k_pX_noise = self.noise_kernel(X_new, self.X_train, params, jitter=0.0)
        k_XX_noise = self.noise_kernel(self.X_train, self.X_train, params, 0, **kwargs)
        # Compute noise predictive mean
        log_var_residual = params["log_var"].copy()
        if self.noise_mean_fn is not None:
            args = [self.X_train, params] if self.noise_mean_fn_prior else [self.X_train]
            log_var_residual -= jnp.log(self.noise_mean_fn(*args)).squeeze()
        K_xx_noise_inv = jnp.linalg.inv(k_XX_noise)
        predicted_log_var = jnp.matmul(k_pX_noise, jnp.matmul(K_xx_noise_inv, log_var_residual))
        if self.noise_mean_fn is not None:
            args = [X_new, params] if self.noise_mean_fn_prior else [X_new]
            predicted_log_var += jnp.log(self.noise_mean_fn(*args)).squeeze()
        predicted_noise_variance = jnp.exp(predicted_log_var)

        # Return the main GP's predictive mean and combined (main + noise) covariance matrix
        return mean, cov + jnp.diag(predicted_noise_variance)

    def get_data_var_samples(self):
        """Returns samples with inferred (training) data variance - aka noise"""
        samples = self.mcmc.get_samples()
        log_var = samples["log_var"]
        if self.noise_mean_fn is not None:
            if self.noise_mean_fn_prior is not None:
                mean_ = jax.vmap(self.noise_mean_fn, in_axes=(None, 0))(self.X_train.squeeze(), samples)
            else:
                mean_ = self.noise_mean_fn(self.X_train.squeeze())
            log_var += jnp.log(mean_)
        return jnp.exp(log_var)

    def _print_summary(self):
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary({k: v for (k, v) in samples.items() if 'log_var' not in k})
