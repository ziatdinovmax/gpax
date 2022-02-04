from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jra
import numpy as onp
import numpyro
import numpyro.distributions as dist
from jax import jit
from numpyro.infer import MCMC, NUTS, init_to_median

from .kernels import get_kernel
from .utils import split_in_batches

if jax.__version__ < '0.2.26':
    clear_cache = jax.interpreters.xla._xla_callable.cache_clear
else:
    clear_cache = jax._src.dispatch._xla_callable.cache_clear


class ExactGP:
    """
    Gaussian process class

    Args:
        input_dim: number of input dimensions
        kernel: type of kernel ('RBF', 'Matern', 'Periodic')
        mean_fn: optional deterministic mean function (use 'mean_fn_priors' to make it probabilistic)
        kernel_prior: optional custom priors over kernel hyperparameters (uses LogNormal(0,1) by default)
        mean_fn_prior: optional priors over mean function parameters
        noise_prior: optional custom prior for observation noise
    """

    def __init__(self, input_dim: int, kernel: str,
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None
                 ) -> None:
        clear_cache()
        self.kernel_dim = input_dim
        self.kernel = kernel
        self.mean_fn = mean_fn
        self.kernel_prior = kernel_prior
        self.mean_fn_prior = mean_fn_prior
        self.noise_prior = noise_prior
        self.X_train = None
        self.y_train = None
        self.mcmc = None

    def model(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """GP probabilistic model"""
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[:2])
        # Sample kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params(task_dim=X.shape[0])
        # Sample noise
        with numpyro.plate("obs_noise", X.shape[0]):
            if self.noise_prior:
                noise = self.noise_prior()
            else:
                noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
        # Add mean function (if any)
        if self.mean_fn is not None:
            args = [X]
            if self.mean_fn_prior is not None:
                args += [self.mean_fn_prior()]
            f_loc += self.mean_fn(*args).squeeze()
        # Compute kernel(s)
        k_args = (X, X, kernel_params, noise)
        k = jax.jit(jax.vmap(get_kernel(self.kernel)))(*k_args)
        # sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def fit(self, rng_key: jnp.array, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000, num_chains: int = 1,
            progress_bar: bool = True, print_summary: bool = True) -> None:
        """
        Run MCMC to infer the GP model parameters

        Args:
            rng_key: random number generator key
            X: 2D 'feature vector' with :math:`n x num_features` dimensions
            y: 1D 'target vector' with :math:`(n,)` dimensions
            num_warmup: number of MCMC warmup states
            num_samples: number of MCMC samples
            num_chains: number of MCMC chains
            progress_bar: show progress bar
            print_summary: print summary at the end of sampling
        """
        X, y = self._set_data(X, y)
        self.X_train = X
        self.y_train = y

        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self.model, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
            jit_model_args=False
        )
        self.mcmc.run(rng_key, X, y)
        if print_summary:
            self._print_summary()

    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def _get_mvn_posterior(self,
                           X_train: jnp.ndarray, y_train: jnp.ndarray,
                           X_new: jnp.ndarray, params: Dict[str, jnp.ndarray]
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        noise = params["noise"]
        y_residual = y_train
        if self.mean_fn is not None:
            args = [self.X_train, params] if self.mean_fn_prior else [self.X_train]
            y_residual -= self.mean_fn(*args).squeeze()
        # compute kernel matrices for train and test data
        k_pp = get_kernel(self.kernel)(X_new, X_new, params, noise)
        k_pX = get_kernel(self.kernel)(X_new, X_train, params, jitter=0.0)
        k_XX = get_kernel(self.kernel)(X_train, X_train, params, noise)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))
        if self.mean_fn is not None:
            args = [X_new, params] if self.mean_fn_prior else [X_new]
            mean += self.mean_fn(*args).squeeze()
        return mean, cov

    @partial(jit, static_argnames='self')
    def get_mvn_posterior(self,
                          X_new: jnp.ndarray, params: Dict[str, jnp.ndarray]
                          ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns parameters (mean and cov) of multivariate normal posterior
        for a single sample of GP hyperparameters. Wrapper over self._get_mvn_posterior.
        """
        params_unsqueezed = {
            k: p[None] if p.ndim == 0 else p for (k,p) in params.items()
        }
        vmap_args = (self.X_train, self.y_train, X_new, params_unsqueezed)
        mean, cov = jax.vmap(self._get_mvn_posterior)(*vmap_args)
        return mean, cov

    def _predict(self, rng_key: jnp.ndarray, X_new: jnp.ndarray,
                 params: Dict[str, jnp.ndarray], n: int
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prediction with a single sample of GP hyperparameters"""
        X_new = self._set_data(X_new)
        # Get the predictive mean and covariance
        y_mean, K = self.get_mvn_posterior(X_new, params)
        # draw samples from the posterior predictive for a given set of hyperparameters
        y_sample = dist.MultivariateNormal(y_mean, K).sample(rng_key, sample_shape=(n,))
        return y_mean, y_sample.squeeze()

    def _sample_kernel_params(self, task_dim: int = None) -> Dict[str, jnp.ndarray]:
        """
        Sample kernel parameters with default
        weakly-informative log-normal priors
        """
        with numpyro.plate("plate_1", task_dim, dim=-2):  # task dimension'
            with numpyro.plate('lengthscale', self.kernel_dim, dim=-1):  # allows using ARD kernel for dim > 1
                length = numpyro.sample("k_length", dist.LogNormal(0.0, 1.0))
        with numpyro.plate("plate_2", task_dim):  # batch/task dimension'
            scale = numpyro.sample("k_scale", dist.LogNormal(0.0, 1.0))
            if self.kernel == 'Periodic':
                period = numpyro.sample("period", dist.LogNormal(0.0, 1.0))
        kernel_params = {
            "k_length": length, "k_scale": scale,
            "period": period if self.kernel == "Periodic" else None}
        return kernel_params

    def predict_in_batches(self, rng_key: jnp.ndarray,
                           X_new: jnp.ndarray,  batch_size: int = 100,
                           samples: Optional[Dict[str, jnp.ndarray]] = None,
                           n: int = 1, filter_nans: bool = False
                           ) -> Tuple[onp.ndarray, onp.ndarray]:
        """
        Make prediction at X_new with sampled GP hyperparameters
        by spitting the input array into chunks ("batches") and running
        self.predict on each of them one-by-one to avoid memory overflow
        """

        def predict_batch(Xi):
            mean, sampled = self.predict(rng_key, Xi, samples, n, filter_nans)
            mean = jax.device_put(mean, jax.devices("cpu")[0])
            sampled = jax.device_put(sampled, jax.devices("cpu")[0])
            if Xi.shape[1] == 1:
                mean, sampled = mean[..., None], sampled[..., None]
            return mean, sampled

        X_new = self._set_data(X_new)
        y_pred, y_sampled = [], []
        for Xi in split_in_batches(X_new, batch_size, dim=1):
            mean, sampled = predict_batch(Xi)
            y_pred.append(mean)
            y_sampled.append(sampled)
        y_pred = onp.concatenate(y_pred, -1)
        y_sampled = onp.concatenate(y_sampled, -1)
        return y_pred, y_sampled

    def predict(self, rng_key: jnp.ndarray, X_new: jnp.ndarray,
                samples: Optional[Dict[str, jnp.ndarray]] = None,
                n: int = 1, filter_nans: bool = False
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new points using sampled GP hyperparameters

        Args:
            rng_key: random number generator key
            X_new: 2D vector with new/'test' data of :math:`n x num_features` dimensionality
            samples: optional posterior samples
            n: number of samples from Multivariate Normal posterior for each MCMC sample with GP hyperaparameters
            filter_nans: filter out samples containing NaN values (if any)

        Returns:
            Center of the mass of sampled means and all the sampled predictions
        """
        if samples is None:
            samples = self.get_samples(chain_dim=False)
        num_samples = samples["k_length"].shape[0]
        vmap_args = (jra.split(rng_key, num_samples), samples)
        predictive = jax.vmap(
            lambda params: self._predict(params[0], X_new, params[1], n))
        y_means, y_sampled = predictive(vmap_args)
        if filter_nans:
            y_sampled_ = [y_i for y_i in y_sampled if not jnp.isnan(y_i).any()]
            y_sampled = jnp.array(y_sampled_)
        return y_means.mean(0).squeeze(), y_sampled

    def _set_data(self,
                  X: jnp.ndarray,
                  y: Optional[jnp.ndarray] = None
                  ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X[:, None] if X.ndim == 1 else X  # add feature pseudo-dimension
        X = X[None] if X.ndim == 2 else X  # add task pseudo-dimension
        if y is not None:
            y = y[None] if y.ndim == 1 else y  # add task pseudo-dimension
            return X, y
        return X

    def _print_summary(self):
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary(samples)
