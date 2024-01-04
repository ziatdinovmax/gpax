"""
sigp.py
=======

Fully Bayesian implementation of Gaussian process regression with uncertain (stochastic) inputs

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from typing import Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from . import ExactGP

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray]


class siGP(ExactGP):

    def __init__(self,
                 input_dim: int,
                 kernel: Union[str, kernel_fn_type],
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior_dist: Optional[dist.Distribution] = None,
                 lengthscale_prior_dist: Optional[dist.Distribution] = None,
                 sigma_x_prior_dist: Optional[dist.Distribution] = None
                 ) -> None:
        args = (input_dim, kernel, mean_fn, kernel_prior, mean_fn_prior, noise_prior, noise_prior_dist, lengthscale_prior_dist)
        super(siGP, self).__init__(*args)
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

    def _sample_x(self, X):
        if self.sigma_x_prior_dist is not None:
            sigma_x_dist = self.sigma_x_prior_dist
        else:
            sigma_x_dist = dist.HalfNormal(1)
        sigma_x = numpyro.sample("sigma_x", sigma_x_dist)
        return numpyro.sample("X_prime", dist.Normal(X, sigma_x))

    def get_mvn_posterior(
        self, X_new: jnp.ndarray, params: Dict[str, jnp.ndarray], noiseless: bool = False, **kwargs: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns parameters (mean and cov) of multivariate normal posterior
        for a single sample of GP parameters
        """
        X_train_prime = params["X_prime"]
        noise = params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        y_residual = self.y_train.copy()
        if self.mean_fn is not None:
            args = [X_train_prime, params] if self.mean_fn_prior else [X_train_prime]
            y_residual -= self.mean_fn(*args).squeeze()
        # compute kernel matrices for train and test data
        k_pp = self.kernel(X_new, X_new, params, noise_p, **kwargs)
        k_pX = self.kernel(X_new, X_train_prime, params, jitter=0.0)
        k_XX = self.kernel(X_train_prime, X_train_prime, params, noise, **kwargs)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))
        if self.mean_fn is not None:
            args = [X_new, params] if self.mean_fn_prior else [X_new]
            mean += self.mean_fn(*args).squeeze()
        return mean, cov

    def _predict(
        self,
        rng_key: jnp.ndarray,
        X_new: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        n: int,
        noiseless: bool = False,
        **kwargs: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prediction with a single sample of GP parameters"""
        # Sample X_new using the learned standard deviation
        X_new_prime = dist.Normal(X_new, params["sigma_x"]).sample(rng_key, sample_shape=(n,))
        X_new_prime = X_new_prime.mean(0)
        # Get the predictive mean and covariance
        y_mean, K = self.get_mvn_posterior(X_new_prime, params, noiseless, **kwargs)
        # draw samples from the posterior predictive for a given set of parameters
        y_sampled = dist.MultivariateNormal(y_mean, K).sample(rng_key, sample_shape=(n,))
        return y_mean, y_sampled

    def _print_summary(self):
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary({k: v for (k, v) in samples.items() if 'X_prime' not in k})
