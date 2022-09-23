"""
gp.py
=======

Fully Bayesian implementation of Gaussian process regression

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from functools import partial
from typing import Callable, Dict, Optional, Tuple, Type, Union

import jax
import jaxlib
import jax.numpy as jnp
import jax.random as jra
import numpyro
import numpyro.distributions as dist
from jax import jit
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive

from .kernels import get_kernel
from .utils import split_in_batches


if jax.__version__ < '0.2.26':
    clear_cache = jax.interpreters.xla._xla_callable.cache_clear
else:
    try:
        clear_cache = jax._src.dispatch._xla_callable.cache_clear
    except AttributeError:
        clear_cache = jax._src.dispatch.xla_callable.cache_clear


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
            Optional custom priors over kernel hyperparameters; uses LogNormal(0,1) by default
        mean_fn_prior:
            Optional priors over mean function parameters
        noise_prior:
            Optional custom prior for observation noise; uses LogNormal(0,1) by default.

    Examples:

        Regular GP for sparse noisy obervations

        >>> # Get random number generator keys for training and prediction
        >>> rng_key, rng_key_predict = gpax.utils.get_keys()
        >>> # Initialize model
        >>> gp_model = gpax.ExactGP(input_dim=1, kernel='Matern')
        >>> # Run HMC to obtain posterior samples for the GP model parameters
        >>> gp_model.fit(rng_key, X, y)  # X and y are arrays with dimensions (n, 1) and (n,)
        >>> # Make a noiseless prediction on new inputs
        >>> y_pred, y_samples = gp_model.predict(rng_key_predict, X_new, noiseless=True)

        GP for noiseless observations

        >>> # Initialize model
        >>> gp_model = gpax.ExactGP(
        >>>     input_dim=1, kernel='RBF',
        >>>     noise_prior = lambda: numpyro.deterministic("noise", 0) # zero observational noise
        >>> )
        >>> # Run HMC to obtain posterior samples for the GP model parameters
        >>> gp_model.fit(rng_key, X, y)  # X and y are arrays with dimensions (n, 1) and (n,)
        >>> # Make prediction on new inputs
        >>> y_pred, y_samples = gp_model.predict(rng_key_predict, X_new)

        GP with custom noise prior
        
        >>> gp_model = gpax.ExactGP(
        >>>     input_dim=1, kernel='RBF',
        >>>     noise_prior = lambda: numpyro.sample("noise", numpyro.distributions.HalfNormal(.1))
        >>> )
        >>> # Run HMC to obtain posterior samples for the GP model parameters
        >>> gp_model.fit(rng_key, X, y)  # X and y are arrays with dimensions (n, 1) and (n,)
        >>> # Make a noiselsess prediction on new inputs
        >>> y_pred, y_samples = gp_model.predict(rng_key_predict, X_new, noiseless=True)

        GP with custom probabilistic model as its mean function
        
        >>> # Define a deterministic mean function
        >>> mean_fn = lambda x, param: param["a"]*x + param["b"]
        >>>
        >>> # Define priors over the mean function parameters (to make it probabilistic)
        >>> def mean_fn_prior():
        >>>     a = numpyro.sample("a", numpyro.distributions.Normal(3, 1))
        >>>     b = numpyro.sample("b", numpyro.distributions.Normal(0, 1))
        >>>     return {"a": a, "b": b}
        >>>
        >>> # Initialize structural GP model
        >>> sgp_model = gpax.ExactGP(
                input_dim=1, kernel='Matern',
                mean_fn=mean_fn, mean_fn_prior=mean_fn_prior)
        >>> # Run HMC to obtain posterior samples for the GP model parameters
        >>> sgp_model.fit(rng_key, X, y)  # X and y are numpy arrays with dimensions (n, d) and (n,)
        >>> # Make a noiselsess prediction on new inputs
        >>> y_pred, y_samples = gp_model.predict(rng_key_predict, X_new, noiseless=True)
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

    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              **kwargs: float
              ) -> None:
        """GP probabilistic model with inputs X and targets y"""
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[0])
        # Sample kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params()
        # Sample noise
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
        # compute kernel
        k = get_kernel(self.kernel)(
            X, X,
            kernel_params,
            noise,
            **kwargs
        )
        # sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def fit(self, rng_key: jnp.array, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            progress_bar: bool = True, print_summary: bool = True,
            device: Type[jaxlib.xla_extension.Device] = None,
            **kwargs: float
            ) -> None:
        """
        Run Hamiltonian Monter Carlo to infer the GP parameters

        Args:
            rng_key: random number generator key
            X: 2D feature vector with *(number of points, number of features)* dimensions
            y: 1D target vector with *(n,)* dimensions
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: 'sequential', 'parallel' or 'vectorized'
            progress_bar: show progress bar
            print_summary: print summary at the end of sampling
            device:
                optionally specify a cpu or gpu device on which to run the inference;
                e.g., ``device=jax.devices("cpu")[0]`` 
            **jitter:
                Small positive term added to the diagonal part of a covariance
                matrix for numerical stability (Default: 1e-6)
        """
        X, y = self._set_data(X, y)
        if device:
            X = jax.device_put(X, device)
            y = jax.device_put(y, device)
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
            jit_model_args=False
        )
        self.mcmc.run(rng_key, X, y, **kwargs)
        if print_summary:
            self._print_summary()

    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    @partial(jit, static_argnames='self')
    def get_mvn_posterior(self,
                          X_new: jnp.ndarray, params: Dict[str, jnp.ndarray],
                          noiseless: bool = False, **kwargs: float
                          ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns parameters (mean and cov) of multivariate normal posterior
        for a single sample of GP parameters
        """
        noise = params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        y_residual = self.y_train
        if self.mean_fn is not None:
            args = [self.X_train, params] if self.mean_fn_prior else [self.X_train]
            y_residual -= self.mean_fn(*args).squeeze()
        # compute kernel matrices for train and test data
        k_pp = get_kernel(self.kernel)(X_new, X_new, params, noise_p, **kwargs)
        k_pX = get_kernel(self.kernel)(X_new, self.X_train, params, jitter=0.0)
        k_XX = get_kernel(self.kernel)(self.X_train, self.X_train, params, noise, **kwargs)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))
        if self.mean_fn is not None:
            args = [X_new, params] if self.mean_fn_prior else [X_new]
            mean += self.mean_fn(*args).squeeze()
        return mean, cov

    def _predict(self, rng_key: jnp.ndarray, X_new: jnp.ndarray,
                 params: Dict[str, jnp.ndarray], n: int, noiseless: bool = False,
                 **kwargs: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prediction with a single sample of GP parameters"""
        # Get the predictive mean and covariance
        y_mean, K = self.get_mvn_posterior(X_new, params, noiseless, **kwargs)
        # draw samples from the posterior predictive for a given set of parameters
        y_sampled = dist.MultivariateNormal(y_mean, K).sample(rng_key, sample_shape=(n,))
        return y_mean, y_sampled

    def _sample_kernel_params(self, dim: int = None) -> Dict[str, jnp.ndarray]:
        """
        Sample kernel parameters with default
        weakly-informative log-normal priors
        """
        with numpyro.plate('k_param', self.kernel_dim):  # allows using ARD kernel for kernel_dim > 1
            length = numpyro.sample("k_length", dist.LogNormal(0.0, 1.0))
        scale = numpyro.sample("k_scale", dist.LogNormal(0.0, 1.0))
        if self.kernel == 'Periodic':
            period = numpyro.sample("period", dist.LogNormal(0.0, 1.0))
        kernel_params = {
            "k_length": length, "k_scale": scale,
            "period": period if self.kernel == "Periodic" else None}
        return kernel_params

    def _predict_in_batches(self, rng_key: jnp.ndarray,
                            X_new: jnp.ndarray,  batch_size: int = 100,
                            batch_dim: int = 0,
                            samples: Optional[Dict[str, jnp.ndarray]] = None,
                            n: int = 1, filter_nans: bool = False,
                            predict_fn: Callable[[jnp.ndarray, int], Tuple[jnp.ndarray]] = None,
                            noiseless: bool = False,
                            device: Type[jaxlib.xla_extension.Device] = None,
                            **kwargs: float
                            ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        if predict_fn is None:
            predict_fn = lambda xi:  self.predict(
                rng_key, xi, samples, n, filter_nans, noiseless, device, **kwargs)

        def predict_batch(Xi):
            out1, out2 = predict_fn(Xi)
            out1 = jax.device_put(out1, jax.devices("cpu")[0])
            out2 = jax.device_put(out2, jax.devices("cpu")[0])
            return out1, out2

        y_out1, y_out2 = [], []
        for Xi in split_in_batches(X_new, batch_size, dim=batch_dim):
            out1, out2 = predict_batch(Xi)
            y_out1.append(out1)
            y_out2.append(out2)
        return y_out1, y_out2

    def predict_in_batches(self, rng_key: jnp.ndarray,
                           X_new: jnp.ndarray,  batch_size: int = 100,
                           samples: Optional[Dict[str, jnp.ndarray]] = None,
                           n: int = 1, filter_nans: bool = False,
                           predict_fn: Callable[[jnp.ndarray, int], Tuple[jnp.ndarray]] = None,
                           noiseless: bool = False,
                           device: Type[jaxlib.xla_extension.Device] = None,
                           **kwargs: float
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new with sampled GP parameters
        by spitting the input array into chunks ("batches") and running
        predict_fn (defaults to self.predict) on each of them one-by-one
        to avoid a memory overflow
        """
        y_pred, y_sampled = self._predict_in_batches(
            rng_key, X_new, batch_size, 0, samples, n,
            filter_nans, predict_fn, noiseless, device, **kwargs)
        y_pred = jnp.concatenate(y_pred, 0)
        y_sampled = jnp.concatenate(y_sampled, -1)
        return y_pred, y_sampled

    def predict(self, rng_key: jnp.ndarray, X_new: jnp.ndarray,
                samples: Optional[Dict[str, jnp.ndarray]] = None,
                n: int = 1, filter_nans: bool = False, noiseless: bool = False,
                device: Type[jaxlib.xla_extension.Device] = None, **kwargs: float
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new points using posterior samples for GP parameters

        Args:
            rng_key: random number generator key
            X_new: new inputs with *(number of points, number of features)* dimensions
            samples: optional (different) samples with GP parameters
            n: number of samples from Multivariate Normal posterior for each HMC sample with GP parameters
            filter_nans: filter out samples containing NaN values (if any)
            noiseless:
                Noise-free prediction. It is set to False by default as new/unseen data is assumed
                to follow the same distribution as the training data. Hence, since we introduce a model noise
                by default for the training data, we also want to include that noise in our prediction.
            device:
                optionally specify a cpu or gpu device on which to make a prediction;
                e.g., ```device=jax.devices("gpu")[0]```
            **jitter:
                Small positive term added to the diagonal part of a covariance
                matrix for numerical stability (Default: 1e-6)

        Returns
            Center of the mass of sampled means and all the sampled predictions
        """
        X_new = self._set_data(X_new)
        if samples is None:
            samples = self.get_samples(chain_dim=False)
        if device:
            self._set_training_data(device=device)
            X_new = jax.device_put(X_new, device)
            samples = jax.device_put(samples, device)
        num_samples = samples["k_length"].shape[0]
        vmap_args = (jra.split(rng_key, num_samples), samples)
        predictive = jax.vmap(
            lambda prms: self._predict(prms[0], X_new, prms[1], n, noiseless, **kwargs))
        y_means, y_sampled = predictive(vmap_args)
        if filter_nans:
            y_sampled_ = [y_i for y_i in y_sampled if not jnp.isnan(y_i).any()]
            y_sampled = jnp.array(y_sampled_)
        return y_means.mean(0), y_sampled

    def sample_from_prior(self, rng_key: jnp.ndarray,
                          X: jnp.ndarray, num_samples: int = 10):
        """
        Samples from prior predictive distribution at X
        """
        X = self._set_data(X)
        prior_predictive = Predictive(self.model, num_samples=num_samples)
        samples = prior_predictive(rng_key, X)
        return samples['y']

    def _set_data(self,
                  X: jnp.ndarray,
                  y: Optional[jnp.ndarray] = None
                  ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            return X, y.squeeze()
        return X

    def _set_training_data(self,
                           X_train_new: jnp.ndarray = None,
                           y_train_new: jnp.ndarray = None,
                           device: Type[jaxlib.xla_extension.Device] = None
                           ) -> None:
        X_train = self.X_train if X_train_new is None else X_train_new
        y_train = self.y_train if y_train_new is None else y_train_new
        if device:
            X_train = jax.device_put(X_train, device)
            y_train = jax.device_put(y_train, device)
        self.X_train = X_train
        self.y_train = y_train

    def _print_summary(self):
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary(samples)
