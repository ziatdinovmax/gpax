"""
spm.py
=======

Bayesian inference with structured probabilistic model. Serves as a wrapper
over the NumPyro's functions for Bayesian inference on probabilistic models.
While it has no direct connection to Gaussian processes, this module may come
in handy for comparisons between predicitons of GP and classical Bayesian models.

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import warnings
from typing import Callable, Optional, Tuple, Type, Dict

import jax
import jaxlib
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

model_type = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]
prior_type = Callable[[], Dict[str, jnp.ndarray]]


class sPM:
    """
    Bayesian inference with structured probabilistic model.
    Serves as a wrapper over the NumPyro's functions for Bayesian inference
    on probabilistic models.

    Args:
        model:
            Deterministic model of expected system's behavior.
        model_prior:
            Priors over model parameters
        noise_prior:
            Optional custom prior for observation noise;
            uses LogNormal(0,1) by default.
    """
    def __init__(self,
                 model: model_type,
                 model_prior: prior_type,
                 noise_prior: Optional[prior_type] = None,
                 noise_prior_dist: Optional[dist.Distribution] = None,) -> None:
        self._model = model
        self.model_prior = model_prior
        if noise_prior is not None:
            warnings.warn(
                "`noise_prior` is deprecated and will be removed in a future version. "
                "Please use `noise_prior_dist` instead, which accepts an instance of a "
                "numpyro.distributions Distribution object, e.g., `dist.HalfNormal(scale=0.1)`, "
                "rather than a function that calls `numpyro.sample`.",
                FutureWarning,
            )
        self.noise_prior = noise_prior
        self.noise_prior_dist = noise_prior_dist
        self.mcmc = None

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None) -> None:
        """
        Full probabilistic model
        """
        # Sample model parameters
        params = self.model_prior()
        # Compute the function's value
        mu = numpyro.deterministic("mu", self._model(X, params))
        # Sample observational noise
        if self.noise_prior:  # this will be removed in the future releases
            sig = self.noise_prior()
        else:
            sig = self._sample_noise()
        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def _sample_noise(self) -> jnp.ndarray:
        if self.noise_prior_dist is not None:
            noise_dist = self.noise_prior_dist
        else:
            noise_dist = dist.LogNormal(0, 1)
        return numpyro.sample("noise", noise_dist)

    def fit(self, rng_key: jnp.array, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            progress_bar: bool = True, print_summary: bool = True,
            device: Type[jaxlib.xla_extension.Device] = None) -> None:
        """
        Run HMC to infer parameters of the structured probabilistic model

        Args:
            rng_key: random number generator key
            X: 1D or 2D 'feature vector' with :math:`(n,)` or :math:`n x num_features` dimensions
            y: 1D 'target vector' with :math:`(n,)` dimensions
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: 'sequential', 'parallel' or 'vectorized'
            progress_bar: show progress bar
            print_summary: print summary at the end of sampling
            device:
                optionally specify a cpu or gpu device on which to run the inference;
                e.g., ``device=jax.devices("cpu")[0]`` 
        """
        if device:
            X = jax.device_put(X, device)
            y = jax.device_put(y, device)
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
        self.mcmc.run(rng_key, X, y)
        if print_summary:
            self._print_summary()

    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def get_param_means(self):
        """
        Returns mean value for each probabilistic parameter in the model
        """
        samples = self.get_samples()
        param_means = {
            k: v.mean(0).item() for (k, v) in samples.items() if k != 'mu'
        }
        return param_means

    def sample_from_prior(self, rng_key: jnp.ndarray,
                          X: jnp.ndarray, num_samples: int = 10):
        """
        Samples from prior predictive distribution at X
        """
        prior_predictive = Predictive(self.model, num_samples=num_samples)
        samples = prior_predictive(rng_key, X)
        return samples['y']

    def predict(self, rng_key: jnp.ndarray, X_new: jnp.ndarray,
                samples: Optional[Dict[str, jnp.ndarray]] = None,
                filter_nans: bool = False, take_point_predictions_mean: bool = True,
                device: Type[jaxlib.xla_extension.Device] = None
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new points using sampled GP hyperparameters

        Args:
            rng_key: random number generator key
            X_new: 2D vector with new/'test' data of :math:`n x num_features` dimensionality
            samples: optional posterior samples
            filter_nans: filter out samples containing NaN values (if any)
            take_point_predictions_mean: take a mean of point predictions (without sampling from the normal distribution)
            device:
                optionally specify a cpu or gpu device on which to make a prediction;
                e.g., ```device=jax.devices("gpu")[0]```

        Returns:
            Point predictions (or their mean) and posterior predictive distribution
        """
        if samples is None:
            samples = self.get_samples(chain_dim=False)
        if device:
            X_new = jax.device_put(X_new, device)
            samples = jax.device_put(samples, device)
        predictive = Predictive(
            self.model, posterior_samples=samples, parallel=True)
        y_pred = predictive(rng_key, X_new)
        y_pred, y_sampled = y_pred["mu"], y_pred["y"]
        if filter_nans:
            y_sampled_ = [y_i for y_i in y_sampled if not jnp.isnan(y_i).any()]
            y_sampled = jnp.array(y_sampled_)
        if take_point_predictions_mean:
            y_pred = y_pred.mean(0)
        return y_pred, y_sampled

    def _print_summary(self):
        self.mcmc.print_summary()
