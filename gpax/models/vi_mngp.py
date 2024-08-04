"""
vi_mngp.py
==========

Variational Gaussian Process model that incorporates measured noise.

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from typing import Callable, Dict, Optional, Union


import jax.numpy as jnp
import jax.random as jra
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta


from .mngp import MeasuredNoiseGP
from ..utils import put_on_device

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray]


class viMeasuredNoiseGP(MeasuredNoiseGP):
    """
    Gaussian Process model that incorporates measured noise.
    This class extends the ExactGP model by allowing the inclusion of measured noise variances
    in the GP framework. Unlike standard GP models where noise is typically inferred, this model
    uses noise values obtained from repeated measurements at the same input points.

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
        lengthscale_prior_dist:
            Optional custom prior distribution over kernel lengthscale. Defaults to LogNormal(0, 1).

    Examples:

        >>> # Initialize model
        >>> gp_model = gpax.viMeasuredNoiseGP(input_dim=1, kernel='Matern')
        >>> # Run SVI to obtain posterior samples for the GP model parameters
        >>> gp_model.fit(X, y_mean, noise)  # X, y_mean, and noise have dimensions (n, 1), (n,), and (n,)
        >>> # Make a prediction on new inputs by extrapolating noise variance with either linear regression or gaussian process
        >>> y_pred, y_samples = gp_model.predict(X_new, noise_prediction_method='linreg')
    """
    def __init__(self,
                 input_dim: int,
                 kernel: Union[str, kernel_fn_type],
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 lengthscale_prior_dist: Optional[dist.Distribution] = None, jitter: float = 1e-6
                 ) -> None:
        args = (input_dim, kernel, mean_fn, kernel_prior, mean_fn_prior, lengthscale_prior_dist, jitter)
        super(viMeasuredNoiseGP, self).__init__(*args)

    def fit(self,
            X: jnp.ndarray, y: jnp.ndarray, measured_noise: jnp.ndarray,
            num_steps: int = 1000, step_size: float = 5e-3,
            progress_bar: bool = True,
            print_summary: bool = True,
            device: str = None,
            rng_key: jnp.array = None,
            ) -> None:
        """
        Run variational inference to learn GP (hyper)parameters

        Args:
            X: 2D feature vector with *(number of points, number of features)* dimensions
            y: 1D target vector with *(n,)* dimensions
            num_steps: number of SVI steps
            step_size: step size schedule for Adam optimizer
            progress_bar: show progress bar
            print_summary: print summary at the end of training
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key: random number generator key
        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X, y = self.set_data(X, y)
        X, y, measured_noise = put_on_device(device, X, y, measured_noise)
        self.X_train = X
        self.y_train = y
        self.measured_noise = measured_noise

        optim = numpyro.optim.Adam(step_size=step_size, b1=0.5)
        self.svi = SVI(
            self.model,
            guide=AutoDelta(self.model),
            optim=optim,
            loss=Trace_ELBO(),
            X=X,
            y=y,
            measured_noise=measured_noise,
        )

        params = self.svi.run(
            key, num_steps, progress_bar=progress_bar)[0]

        self.params = self.svi.guide.median(params)

        if print_summary:
            self.print_summary()

    def get_samples(self, **kwargs):
        samples = {k: v[None] for (k, v) in self.params.items()}
        samples["noise"] = jnp.array([0.0])
        return samples

    def print_summary(self) -> None:
        for (k, vals) in self.params.items():
            spaces = " " * (15 - len(k))
            print(k, spaces, jnp.around(vals, 4))
