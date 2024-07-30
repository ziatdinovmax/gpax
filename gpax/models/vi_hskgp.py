"""
hskgp.py
=========

Fully Bayesian implementation of heteroskedastic Gaussian process regression

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from typing import Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jra

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

from . import VarNoiseGP
from ..kernels import get_kernel
from ..utils import _set_noise_kernel_fn, put_on_device

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray]


class viVarNoiseGP(VarNoiseGP):
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

        >>> # Initialize model
        >>> gp_model = gpax.VarNoiseGP(input_dim=1, kernel='RBF, noise_kernel='Matern')
        >>> # Run HMC to obtain posterior samples for the GP model parameters
        >>> gp_model.fit(X, y)
        >>> # Make a prediction on new inputs
        >>> y_pred, y_samples = gp_model.predict(X_new)
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
        >>> gp_model.fit(X, y)
        >>> # Make a prediction on new inputs
        >>> y_pred, y_samples = gp_model.predict(X_new)
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
        noise_lengthscale_prior_dist: Optional[dist.Distribution] = None,
        jitter: float = 1e-6
    ) -> None:
        super(viVarNoiseGP, self).__init__(input_dim, kernel, noise_kernel, mean_fn, kernel_prior, mean_fn_prior,
                                           noise_kernel_prior, lengthscale_prior_dist, noise_mean_fn, noise_mean_fn_prior,
                                           noise_lengthscale_prior_dist, jitter)

    def fit(self,
            X: jnp.ndarray, y: jnp.ndarray,
            num_steps: int = 1000, step_size: float = 5e-3,
            progress_bar: bool = True,
            device: str = None,
            rng_key: jnp.array = None,
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
            rng_key: random number generator key
        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X, y = self.set_data(X, y)
        X, y = put_on_device(device, X, y)
        self.X_train = X
        self.y_train = y

        optim = numpyro.optim.Adam(step_size=step_size, b1=0.5)
        self.svi = SVI(
            self.model,
            guide=AutoDelta(self.model),
            optim=optim,
            loss=Trace_ELBO(),
            X=X,
            y=y,
            **kwargs
        )

        params = self.svi.run(
            key, num_steps, progress_bar=progress_bar)[0]

        self.params = self.svi.guide.median(params)

    def get_samples(self, **kwargs):
        return {k: v[None] for (k, v) in self.params.items()}
    
    def print_summary(self):
        for (k, vals) in self.params.items():
            if 'log_var' not in k:
                spaces = " " * (15 - len(k))
                print(k, spaces, jnp.around(vals, 4))
