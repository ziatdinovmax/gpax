from typing import Dict, Tuple, Optional, Union, List, Type
import jax.random as jra
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive
from numpyro.contrib.module import random_haiku_module

import haiku as hk

from .nets import HaikuMLP
from ..utils import put_on_device, split_dict


class BNN:
    """
    A Fully Bayesian Neural Network.
    This approach employs a probabilistic treatment of all neural network weights,
    treating them as random variables with specified prior distributions
    and utilizing advanced Markov Chain Monte Carlo techniques to sample directly
    from the posterior distribution, allowing to account for all plausible weight configurations.
    This approach enables the network to make probabilistic predictions,
    not just single-point estimates but entire distributions of possible outcomes,
    quantifying the inherent uncertainty.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: List[int] = None,
                 activation: str = 'tanh',
                 noise_prior: Optional[dist.Distribution] = None,
                 nn: Type[hk.Module] = None
                 ) -> None:
        if noise_prior is None:
            noise_prior = dist.HalfNormal(1.0)
        if nn is not None:
            self.nn_module = hk.transform(lambda x: nn()(x))
        else:
            hdim = hidden_dim if hidden_dim is not None else [32, 16, 8]
            self.nn_module = hk.transform(lambda x: HaikuMLP(hdim, output_dim, activation)(x))
        self.data_dim = (input_dim,) if isinstance(input_dim, int) else input_dim
        self.noise_prior = noise_prior

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """BNN probabilistic model"""

        net = random_haiku_module(
                "feature_extractor", self.nn_module, input_shape=(1, *self.data_dim),
                prior=(lambda name, shape: dist.Cauchy() if name.startswith("b") else dist.Normal()))

        # Pass inputs through a NN with the sampled parameters
        mu = numpyro.deterministic("mu", net(X))

        # Sample noise
        sig = self.sample_noise()

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            progress_bar: bool = True, device: str = None,
            rng_key: Optional[jnp.array] = None,
            ) -> None:
        """
        Run HMC to infer parameters of the BNN

        Args:
            X: 2D feature vector
            y: 1D target vector
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: 'sequential', 'parallel' or 'vectorized'
            progress_bar: show progress bar
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key: random number generator key
        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X, y = self.set_data(X, y)
        X, y = put_on_device(device, X, y)
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
        self.mcmc.run(key, X, y)

    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def sample_noise(self) -> jnp.ndarray:
        """
        Sample observational noise variance
        """
        return numpyro.sample("sig", self.noise_prior)

    def predict(self,
                X_new: jnp.ndarray,
                samples: Optional[Dict[str, jnp.ndarray]] = None,
                device: Optional[str] = None,
                rng_key: Optional[jnp.ndarray] = None
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict the mean and variance of the target values for new inputs.

        Args:
            X_new:
                New input data for predictions.
            samples:
                Dictionary of posterior samples with inferred model parameters (weights and biases)
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key:
                Random number generator key for JAX operations.

        Returns:
            Tuple containing the means and samples from the posterior predictive distribution.
        """
        X_new = self.set_data(X_new)

        if rng_key is None:
            rng_key = jra.PRNGKey(0)
        if samples is None:
            samples = self.get_samples(chain_dim=False)
        X_new, samples = put_on_device(device, X_new, samples)

        predictions = self.sample_from_posterior(
            rng_key, X_new, samples, return_sites=["mu", "y"])
        posterior_mean = predictions["mu"].mean(0)
        posterior_var = predictions["y"].var(0)
        return posterior_mean, posterior_var

    def sample_from_posterior(self,
                              rng_key: jnp.ndarray,
                              X_new: jnp.ndarray,
                              samples: Dict[str, jnp.ndarray],
                              return_sites: Optional[List[str]] = None,
                              ) -> jnp.ndarray:
   
        predictive = Predictive(
            self.model, samples,
            return_sites=return_sites
        )
        return predictive(rng_key, X_new)

    # def predict_in_batches(self, X_new: jnp.ndarray,
    #                        batch_size: int = 100,
    #                        n_draws: int = 1,
    #                        device: Optional[str] = None,
    #                        rng_key: Optional[jnp.ndarray] = None
    #                        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     """
    #     Make prediction in batches (to avoid memory overflow)
    #     at X_new points a trained BNN model
    #     """
    #     samples = self.get_samples(chain_dim=False)
    #     mean_chunks, f_samples_chunks = [], []
    #     for batch in split_dict(samples, batch_size):
    #         mean_i, f_samples_i = self._vmap_predict(X_new, batch, n_draws, rng_key, device)
    #         mean_i = jax.device_put(mean_i, jax.devices("cpu")[0])
    #         f_samples_i = jax.device_put(f_samples_i, jax.devices("cpu")[0])
    #         mean_chunks.append(mean_i[None])
    #         f_samples_chunks.append(f_samples_i)
    #     mean_chunks = jnp.concatenate(mean_chunks, axis=0)
    #     f_samples_chunks = jnp.concatenate(f_samples_chunks)
        
    #     return mean_chunks.mean(0), f_samples_chunks.var(0)

    def set_data(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None
                 ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            y = y[:, None] if y.ndim < 2 else y
            return X, y
        return X