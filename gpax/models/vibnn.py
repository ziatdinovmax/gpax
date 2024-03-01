"""
vibnn.py
========

Variational inference-based implementation of Bayesian MLP

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from typing import Callable, Dict, Optional, Tuple, List, Type, Union

import jax
import jaxlib
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta, AutoNormal


class viBNN():
    """
    Implementation of the variational inference-based Bayesian neural network
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 noise_prior_dist: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 hidden_dim: Optional[List[int]] = None, guide: str = 'normal', **kwargs
                 ) -> None:

        if guide not in ['delta', 'normal']:
            raise NotImplementedError("Select guide between 'delta' and 'normal'")
        hidden_dim = [64, 32] if not hidden_dim else hidden_dim
        self.nn = kwargs.get("nn", get_mlp(hidden_dim))
        self.nn_prior = kwargs.get("nn_prior", get_mlp_prior(input_dim, output_dim, hidden_dim))
        self.noise_prior_dist = noise_prior_dist
        self.guide_type = AutoDelta if guide == 'delta' else AutoNormal
        self.svi = None

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """Probabilistic model"""

        # Sample NN parameters
        nn_params = self.nn_prior()
        # Pass inputs through a NN with sampled parameters
        mu = numpyro.deterministic("mu", self.nn(X, nn_params))

        # Sample noise
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
            num_steps: int = 1000, step_size: float = 5e-3,
            progress_bar: bool = True,
            device: Type[jaxlib.xla_extension.Device] = None,
            **kwargs: float
            ) -> None:
        """
        Run variational inference to learn GP (hyper)parameters

        Args:
            rng_key: random number generator key
            X: 2D feature vector with *(number of points, number of features)* dimensions
            y: 1D target vector with *(n,)* dimensions
            num_steps: number of SVI steps
            step_size: step size schedule for Adam optimizer
            progress_bar: show progress bar
            print_summary: print summary at the end of training
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

        optim = numpyro.optim.Adam(step_size=step_size, b1=0.5)
        self.svi = SVI(
            self.model,
            guide=self.guide_type(self.model),
            optim=optim,
            loss=Trace_ELBO(),
            X=X,
            y=y,
            **kwargs
        )

        self.nn_params = self.svi.run(
            rng_key, num_steps, progress_bar=progress_bar)[0]


    def get_samples(self) -> Dict[str, jnp.ndarray]:
        """Get posterior samples"""
        return self.svi.guide.median(self.nn_params)

    def predict(self, rng_key: jnp.ndarray, X_new: jnp.ndarray, num_samples: int = 1000,
                device: Type[jaxlib.xla_extension.Device] = None, filter_nans: bool = False
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new points using posterior samples for GP parameters

        Args:
            rng_key: random number generator key
            X_new: new inputs with *(number of points, number of features)* dimensions
            device:
                optionally specify a cpu or gpu device on which to make a prediction;
                e.g., ```device=jax.devices("gpu")[0]```

        Returns
            Center of the mass of sampled means and all the sampled predictions
        """
        X_new = self._set_data(X_new)
        if device:
            X_new = jax.device_put(X_new, device)

        predictive = Predictive(
            self.model, guide=self.svi.guide,
            params=self.nn_params, num_samples=num_samples)

        y_pred = predictive(rng_key, X_new)
        y_pred, y_sampled = y_pred["mu"], y_pred["y"]
        if filter_nans:
            y_sampled_ = [y_i for y_i in y_sampled if not jnp.isnan(y_i).any()]
            y_sampled = jnp.array(y_sampled_)

        return y_pred, y_sampled

    def _set_data(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None
                  ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            y = y[:, None] if y.ndim < 1 else y
            return X, y
        return X


def sample_weights(name: str, in_channels: int, out_channels: int) -> jnp.ndarray:
    """Sampling weights matrix"""
    w = numpyro.sample(name=name, fn=dist.Normal(
        loc=jnp.zeros((in_channels, out_channels)),
        scale=jnp.ones((in_channels, out_channels))))
    return w


def sample_biases(name: str, channels: int) -> jnp.ndarray:
    """Sampling bias vector"""
    b = numpyro.sample(name=name, fn=dist.Normal(
        loc=jnp.zeros((channels)), scale=jnp.ones((channels))))
    return b


def get_mlp(architecture: List[int]) -> Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]:
    """Returns a function that represents an MLP for a given architecture."""
    def mlp(X: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """MLP for a single MCMC sample of weights and biases, handling arbitrary number of layers."""
        h = X
        for i in range(len(architecture)):
            h = jnp.tanh(jnp.matmul(h, params[f"w{i}"]) + params[f"b{i}"])
        # No non-linearity after the last layer
        z = jnp.matmul(h, params[f"w{len(architecture)}"]) + params[f"b{len(architecture)}"]
        return z
    return mlp


def get_mlp_prior(input_dim: int, output_dim: int, architecture: List[int]) -> Callable[[], Dict[str, jnp.ndarray]]:
    """Priors over weights and biases for a Bayesian MLP"""
    def mlp_prior():
        params = {}
        in_channels = input_dim
        for i, out_channels in enumerate(architecture):
            params[f"w{i}"] = sample_weights(f"w{i}", in_channels, out_channels)
            params[f"b{i}"] = sample_biases(f"b{i}", out_channels)
            in_channels = out_channels
        # Output layer
        params[f"w{len(architecture)}"] = sample_weights(f"w{len(architecture)}", in_channels, output_dim)
        params[f"b{len(architecture)}"] = sample_biases(f"b{len(architecture)}", output_dim)
        return params
    return mlp_prior
