"""
uigp.py
=======

Fully Bayesian implementation of Gaussian process regression with uncertain (stochastic) inputs

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jra
import numpyro
import numpyro.distributions as dist

from . import ExactGP

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray]


class UIGP(ExactGP):
    """
    Gaussian process with uncertain inputs

    This class extends the standard Gaussian Process model to handle uncertain inputs.
    It allows for incorporating the uncertainty in input data into the GP model, providing
    a more robust prediction.

    Args:
        input_dim:
            Number of input dimensions
        kernel:
            Kernel function ('RBF', 'Matern', 'Periodic', or custom function)
        mean_fn:
            Optional deterministic mean function (use 'mean_fn_priors' to make it probabilistic)
        kernel_prior:
            Optional custom priors over kernel hyperparameters. Use it when passing your custom kernel.
        mean_fn_prior:
            Optional priors over mean function parameters
        noise_prior_dist:
            Optional custom prior distribution over the observational noise variance.
            Defaults to LogNormal(0,1).
        lengthscale_prior_dist:
            Optional custom prior distribution over kernel lengthscale. Defaults to LogNormal(0, 1).
        sigma_x_prior_dist:
            Optional custom prior for the input uncertainty (sigma_x). Defaults to HalfNormal(0.1)
            under the assumption that data is normalized to (0, 1).

    Examples:

        UIGP with custom prior over sigma_x

        >>> # Get random number generator keys for training and prediction
        >>> rng_key, rng_key_predict = gpax.utils.get_keys()
        >>> # Initialize model
        >>> gp_model = gpax.UIGP(input_dim=1, kernel='Matern', sigma_x_prior_dist=gpax.utils.halfnormal_dist(0.5))
        >>> # Run HMC to obtain posterior samples for the model parameters
        >>> gp_model.fit(rng_key, X, y, num_warmup=2000, num_samples=10000)
        >>> # Make a prediction on new inputs
        >>> y_pred, y_samples = gp_model.predict(rng_key_predict, X_new)
    """
    def __init__(self,
                 input_dim: int,
                 kernel: Union[str, kernel_fn_type],
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior_dist: Optional[dist.Distribution] = None,
                 lengthscale_prior_dist: Optional[dist.Distribution] = None,
                 sigma_x_prior_dist: Optional[dist.Distribution] = None,
                 jitter: float = 1e-6
                 ) -> None:
        args = (input_dim, kernel, mean_fn, kernel_prior, mean_fn_prior,
                noise_prior, noise_prior_dist, lengthscale_prior_dist, jitter)
        super(UIGP, self).__init__(*args)
        self.sigma_x_prior_dist = sigma_x_prior_dist

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs: float) -> None:
        """
        Gaussian process model for uncertain (stochastic) inputs
        """
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[0])

        # Sample input X
        X_prime = self._sample_x(X)

        # Sample kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params()
        # Sample noise
        if self.noise_prior:  # this will be removed in the future releases
            noise = self.noise_prior()
        else:
            noise = self._sample_noise()
        # Add mean function (if any)
        if self.mean_fn is not None:
            args = [X_prime]
            if self.mean_fn_prior is not None:
                args += [self.mean_fn_prior()]
            f_loc += self.mean_fn(*args).squeeze()
        # compute kernel
        k = self.kernel(X_prime, X_prime, kernel_params, noise, **kwargs)
        # sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def _sample_x(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Samples new input values (X_prime) based on the original inputs (X)
        and prior belief about the uncertainty in those inputs.
        """
        n_samples, n_features = X.shape
        if self.sigma_x_prior_dist is not None:
            sigma_x_dist = self.sigma_x_prior_dist
        else:
            sigma_x_dist = dist.HalfNormal(.1 * jnp.ones(n_features))
        # Sample variances independently for each feature dimension
        with numpyro.plate("feature_variance_plate", self.kernel_dim):
            sigma_x = numpyro.sample("sigma_x", sigma_x_dist)
            # Sample input data using the sampled variances
            with numpyro.plate("X_prime_plate", n_samples, dim=-2):
                X_prime = numpyro.sample("X_prime", dist.Normal(X, sigma_x))
        return X_prime

    def compute_gp_posterior(self, X_new: jnp.ndarray,
                             X_train: jnp.ndarray, y_train: jnp.ndarray,
                             params: Dict[str, jnp.ndarray],
                             noiseless: bool = True,
                             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns parameters (mean and cov) of multivariate normal posterior
        for a single sample of UIGP parameters
        """
        X_train_prime = params["X_prime"]
        X_new_prime = dist.Normal(X_new, params["sigma_x"]).sample(jra.PRNGKey(0))
        return super().compute_gp_posterior(X_new_prime, X_train_prime, y_train, params, noiseless)

    def _set_data(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            if not (X.max() == 1 and X.min() == 0) and not self.sigma_x_prior_dist:
                warnings.warn(
                    "The default `sigma_x` prior for uncertain (stochastic) inputs assumes data is "
                    "normalized to (0, 1), which is not the case for your data. Therefore, the default prior "
                    "may not be optimal for your case. Consider passing custom prior for sigma_x, for example, "
                    "`sigma_x_prior_dist=numpyro.distributions.HalfNormal(scale)` if using NumPyro directly "
                    "or `sigma_x_prior_dist=gpax.utils.halfnormal_dist(scale)` if using a GPax wrapper",
                    UserWarning,
                )
            return X, y.squeeze()
        return X

    def _print_summary(self):
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary({k: v for (k, v) in samples.items() if 'X_prime' not in k})
