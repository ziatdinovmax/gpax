from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import numpyro
from jax.interpreters import xla
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

from .gp import ExactGP


class viGP(ExactGP):
    """
    Gaussian process via stochastic variational inference

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
        args = (input_dim, kernel, kernel_prior, mean_fn, mean_fn_prior, noise_prior)
        super(viGP, self).__init__(*args)
        xla._xla_callable.cache_clear()
        self.X_train = None
        self.y_train = None
        self.kernel_params = None

    def fit(self, rng_key: jnp.array, X: jnp.ndarray, y: jnp.ndarray,
            num_steps: int = 1000, print_summary: bool = True) -> None:
        """
        Run SVI to infer the GP model parameters

        Args:
            rng_key: random number generator key
            X: 2D 'feature vector' with :math:`n x num_features` dimensions
            y: 1D 'target vector' with :math:`(n,)` dimensions
            num_steps: number of SVI steps
            print_summary: print summary at the end of sampling
        """
        X = X if X.ndim > 1 else X[:, None]
        self.X_train = X
        self.y_train = y
        # Setup optimizer and SVI
        optim = numpyro.optim.Adam(step_size=0.005, b1=0.5)
        svi = SVI(
            self.model,
            guide=AutoDelta(self.model),
            optim=optim,
            loss=Trace_ELBO(),
            X=X,
            y=y,
        )
        params = svi.run(rng_key, num_steps)[0]
        # Get kernel parameters from the guide
        self.kernel_params = svi.guide.median(params)
        if print_summary:
            self._print_summary()

    def predict(self, rng_key: jnp.ndarray, X_new: jnp.ndarray,
                kernel_params: Optional[Dict[str, jnp.ndarray]] = None,
                n: int = 1000
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new points using learned GP hyperparameters

        Args:
            rng_key: random number generator key
            X_new: 2D vector with new/'test' data of :math:`n x num_features` dimensionality
            samples: kernel posterior parameters (optional)
            n: number of samples from the Multivariate Normal posterior

        Returns:
            Center of the mass of sampled means and all the sampled predictions
        """
        if kernel_params is None:
            kernel_params = self.kernel_params
        y_mean, y_sampled = self._predict(rng_key, X_new, kernel_params, n)
        return y_mean, y_sampled

    def _print_summary(self) -> None:
        if isinstance(self.kernel_params, dict):
            print('\nInferred parameters')
            for (k, v) in self.kernel_params.items():
                spaces = " " * (15 - len(k))
                print(k, spaces, jnp.around(v, 4))

