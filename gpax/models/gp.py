"""
gp.py
=======

Fully Bayesian implementation of Gaussian process regression

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import warnings
from typing import Callable, Dict, Optional, Tuple, Union, Type

import jax
import jaxlib
import jax.numpy as jnp
import jax.random as jra
from jax import vmap
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive

from ..kernels import get_kernel
from ..utils import put_on_device, split_in_batches

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray]

clear_cache = jax._src.dispatch.xla_primitive_callable.cache_clear


class ExactGP:
    """
    Gaussian process class

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
            Optional custom prior distribution over kernel lengthscale.
            Defaults to LogNormal(0, 1).
        jitter:
            Small jitter for the numerical stability. Default: 1e-6


    Examples:

        Regular GP for sparse noisy obervations

        >>> # Initialize model
        >>> gp_model = gpax.ExactGP(input_dim=1, kernel='Matern')
        >>> # Run HMC to obtain posterior samples for the GP model parameters
        >>> gp_model.fit(X, y)  # X and y are arrays with dimensions (n, 1) and (n,)
        >>> # Make a noiseless prediction on new inputs
        >>> y_pred, y_samples = gp_model.predict(X_new, noiseless=True)

        GP with custom noise prior

        >>> gp_model = gpax.ExactGP(
        >>>     input_dim=1, kernel='RBF',
        >>>     noise_prior_dist = gpax.priors.halfnormal_dist(0.1)
        >>> )
        >>> # Run HMC to obtain posterior samples for the GP model parameters
        >>> gp_model.fit(X, y)  # X and y are arrays with dimensions (n, 1) and (n,)
        >>> # Make a noiselsess prediction on new inputs
        >>> y_pred, y_samples = gp_model.predict(X_new, noiseless=True)

        GP with custom probabilistic model as its mean function

        >>> # Define a deterministic mean function
        >>> mean_fn = lambda x, param: param["a"]*x + param["b"]
        >>>
        >>> # Define priors over the mean function parameters (to make it probabilistic)
        >>> def mean_fn_prior():
        >>>     a = gpax.priors.place_normal_prior("a", loc=3, scale=1)
        >>>     b = gpax.priors.place_normal_prior("a", loc=0, scale=1)
        >>>     return {"a": a, "b": b}
        >>>
        >>> # Initialize structured GP model
        >>> sgp_model = gpax.ExactGP(
                input_dim=1, kernel='Matern',
                mean_fn=mean_fn, mean_fn_prior=mean_fn_prior)
        >>> # Run HMC to obtain posterior samples for the GP model parameters
        >>> sgp_model.fit(X, y)  # X and y are numpy arrays with dimensions (n, d) and (n,)
        >>> # Make a noiselsess prediction on new inputs
        >>> y_pred, y_samples = gp_model.predict(X_new, noiseless=True)
    """

    def __init__(
        self,
        input_dim: int,
        kernel: Union[str, kernel_fn_type],
        mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
        kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
        noise_prior_dist: Optional[dist.Distribution] = None,
        lengthscale_prior_dist: Optional[dist.Distribution] = None,
        jitter: float = 1e-6
    ) -> None:
        clear_cache()
        if noise_prior is not None:
            warnings.warn(
                "`noise_prior` is deprecated and will be removed in a future version. "
                "Please use `noise_prior_dist` instead, which accepts an instance of a "
                "numpyro.distributions Distribution object, e.g., `dist.HalfNormal(scale=0.1)`, "
                "rather than a function that calls `numpyro.sample`.",
                FutureWarning,
            )
        if kernel_prior is not None:
            warnings.warn(
                "`kernel_prior` will remain available for complex priors. However, for "
                "modifying only the lengthscales, it is recommended to use `lengthscale_prior_dist` instead. "
                "`lengthscale_prior_dist` accepts an instance of a numpyro.distributions Distribution object, "
                "e.g., `dist.Gamma(2, 5)`, rather than a function that calls `numpyro.sample`.",
                UserWarning,
            )
        self.kernel_dim = input_dim
        self.kernel = get_kernel(kernel)
        self.kernel_name = kernel if isinstance(kernel, str) else None
        self.mean_fn = mean_fn
        self.kernel_prior = kernel_prior
        self.mean_fn_prior = mean_fn_prior
        self.noise_prior = noise_prior
        self.noise_prior_dist = noise_prior_dist
        self.lengthscale_prior_dist = lengthscale_prior_dist
        self.X_train = None
        self.y_train = None
        self.mcmc = None
        self.jitter = jitter

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs: float) -> None:
        """GP probabilistic model with inputs X and targets y"""
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[0])
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
            args = [X]
            if self.mean_fn_prior is not None:
                args += [self.mean_fn_prior()]
            f_loc += self.mean_fn(*args).squeeze()
        # compute kernel
        k = self.kernel(X, X, kernel_params, noise, self.jitter)
        # sample y according to the standard Gaussian process formula
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
            progress_bar: show progress bar
            print_summary: print summary at the end of sampling
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key: random number generator key
        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X, y = self.set_data(X, y)
        X, y = put_on_device(device, X, y)
        self.X_train = X
        self.y_train = y

        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self.model, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            jit_model_args=False,
        )
        self.mcmc.run(key, X, y)

        if print_summary:
            self.print_summary()

    def _sample_noise(self) -> jnp.ndarray:
        if self.noise_prior_dist is not None:
            noise_dist = self.noise_prior_dist
        else:
            noise_dist = dist.LogNormal(0, 1)
        return numpyro.sample("noise", noise_dist)

    def _sample_kernel_params(self, output_scale=True) -> Dict[str, jnp.ndarray]:
        """
        Sample kernel parameters with default
        weakly-informative log-normal priors
        """
        if self.lengthscale_prior_dist is not None:
            length_dist = self.lengthscale_prior_dist
        else:
            length_dist = dist.LogNormal(0.0, 1.0)
        with numpyro.plate("ard", self.kernel_dim):  # allows using ARD kernel for kernel_dim > 1
            length = numpyro.sample("k_length", length_dist)
        if output_scale:
            scale = numpyro.sample("k_scale", dist.LogNormal(0.0, 1.0))
        else:
            scale = numpyro.deterministic("k_scale", jnp.array(1.0))
        if self.kernel_name == "Periodic":
            period = numpyro.sample("period", dist.LogNormal(0.0, 1.0))
        kernel_params = {"k_length": length, "k_scale": scale, "period": period if self.kernel_name == "Periodic" else None}
        return kernel_params

    def compute_gp_posterior(self, X_new: jnp.ndarray,
                             X_train: jnp.ndarray, y_train: jnp.ndarray,
                             params: Dict[str, jnp.ndarray],
                             noiseless: bool = True,
                             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns mean and covariance of multivariate normal
        posterior for a single sample of trained GP parameters
        """
        noise = params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        y_residual = y_train.copy()  # appears to be redundant
        if self.mean_fn is not None:
            args = [X_train, params] if self.mean_fn_prior else [X_train]
            y_residual -= self.mean_fn(*args).squeeze()
        # compute kernel matrices for train and test data
        k_XX = self.kernel(X_train, X_train, params, noise, self.jitter)
        k_pp = self.kernel(X_new, X_new, params, noise_p, self.jitter)
        k_pX = self.kernel(X_new, X_train, params)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))
        if self.mean_fn is not None:
            args = [X_new, params] if self.mean_fn_prior else [X_new]
            mean += self.mean_fn(*args).squeeze()
        return mean, cov

    def predict(self,
                X_new: jnp.ndarray,
                noiseless: bool = True,
                device: str = None
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new points a trained GP model

        Args:
            X_new:
                New inputs with *(number of points, number of features)* dimensions
            noiseless:
                Noise-free prediction. It is set to False by default as new/unseen data is assumed
                to follow the same distribution as the training data. Hence, since we introduce a model noise
                by default for the training data, we also want to include that noise in our prediction.
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.

        Returns:
            Posterior mean and variance
        """
        X_new = self.set_data(X_new)
        samples = self.get_samples(chain_dim=False)
        self.X_train, self.y_train, X_new, samples = put_on_device(
            device, self.X_train, self.y_train, X_new, samples)

        predictive = lambda p: self.compute_gp_posterior(
            X_new, self.X_train, self.y_train, p, noiseless)
        # Compute predictive mean and covariance for all HMC samples
        mu_all, cov_all = vmap(predictive)(samples)
        # Calculate the average of the means
        mean_predictions = mu_all.mean(axis=0)
        # Calculate the average within-model variance and variance of the means
        average_within_model_variance = cov_all.mean(axis=0).diagonal()
        variance_of_means = jnp.var(mu_all, axis=0)
        # Total predictive variance
        total_predictive_variance = average_within_model_variance + variance_of_means

        return mean_predictions, total_predictive_variance

    def predict_in_batches(self, X_new: jnp.ndarray,
                           batch_size: int = 200,
                           noiseless: bool = True,
                           device: str = None
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction in batches (to avoid memory overflow) 
        at X_new points a trained GP model
        """
        mean, var = [], []
        for x in split_in_batches(X_new, batch_size):
            mean_i, var_i = self.predict(x, noiseless, device)
            mean_i = jax.device_put(mean_i, jax.devices("cpu")[0])
            var_i = jax.device_put(var_i, jax.devices("cpu")[0])
            mean.append(mean_i)
            var.append(var_i)
        return jnp.concatenate(mean), jnp.concatenate(var)

    def draw_from_mvn(self,
                      rng_key: jnp.ndarray,
                      X_new: jnp.ndarray,
                      params: Dict[str, jnp.ndarray],
                      n_draws: int,
                      noiseless: bool
                      ) -> jnp.ndarray:
        """
        Draws predictive samples from multivariate normal distribution
        at X_new for a single estimate of GP posterior parameters
        """
        mu, cov = self.compute_gp_posterior(
            X_new, self.X_train, self.y_train, params, noiseless)
        mvn = dist.MultivariateNormal(mu, cov)
        return mvn.sample(rng_key, sample_shape=(n_draws,))

    def sample_from_posterior(self,
                              X_new: jnp.ndarray,
                              noiseless: bool = True,
                              samples: Dict[str, jnp.ndarray] = None,
                              n_draws: int = 100,
                              device: str = None,
                              rng_key: jnp.ndarray = None
                              ) -> jnp.ndarray:
        """
        Sample from the posterior predictive distribution at X_new

        Args:
            X_new:
                New inputs with *(number of points, number of features)* dimensions
            noiseless:
                Noise-free prediction. It is set to False by default as new/unseen data is assumed
                to follow the same distribution as the training data. Hence, since we introduce a model noise
                by default for the training data, we also want to include that noise in our prediction.
            samples:
                Optional samples with model parameters. Uses samples from the last MCMC run by default.
            n_draws:
                Number of MVN distribution samples to draw for each sample with GP parameters
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key:
                Optional random number generator key

        Returns:
            A set of samples from the posterior predictive distribution.

        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X_new = self.set_data(X_new)
        if not samples:
            samples = self.get_samples(chain_dim=False)
        self.X_train, self.y_train, X_new, samples = put_on_device(
            device, self.X_train, self.y_train, X_new, samples)

        num_samples = len(next(iter(samples.values())))
        vmap_args = (jra.split(key, num_samples), samples)
        predictive = lambda p1, p2: self.draw_from_mvn(p1, X_new, p2, n_draws, noiseless)
        return vmap(predictive)(*vmap_args)
    
    def sample_from_prior(self,
                          X: jnp.ndarray,
                          num_samples: int = 10,
                          rng_key: jnp.ndarray = None) -> jnp.ndarray:
        """
        Samples from prior predictive distribution at X
        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X = self.set_data(X)
        prior_predictive = Predictive(self.model, num_samples=num_samples)
        samples = prior_predictive(key, X)
        return samples["y"]

    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def set_data(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None
                 ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            return X, y.squeeze()
        return X

    def print_summary(self) -> None:
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary(samples)