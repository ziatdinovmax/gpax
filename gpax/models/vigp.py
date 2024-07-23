from typing import Dict, Callable, Optional, Union
import jax.numpy as jnp
import jax.random as jra
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

from .gp import ExactGP
from ..utils import put_on_device


kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],  jnp.ndarray]


class viGP(ExactGP):
    """
    Variational inference based Gaussian process

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
        noise_prior_dist:
            Optional custom prior distribution over the observational noise variance.
            Defaults to LogNormal(0,1).
        lengthscale_prior_dist:
            Optional custom prior distribution over kernel lengthscale.
            Defaults to LogNormal(0, 1).
        guide:
            Auto-guide option, use 'delta' (default) or 'normal'
        jitter:
            small jitter for the numerical stability

    Examples:

        Use viGP to reconstruct data from sparse noisy obervations

        >>> # Initialize model
        >>> gp_model = gpax.viGP(input_dim=1, kernel='Matern')
        >>> # Run variational inference to obtain a MAP estimate for the GP model parameters
        >>> gp_model.fit(X, y, num_steps=1000)  # X and y are arrays with dimensions (n, 1) and (n,)
        >>> # Make a noiseless prediction on new inputs
        >>> posterior_mean, posterior_var = gp_model.predict(X_new, noiseless=True)
    """

    def __init__(self,
                 input_dim: int,
                 kernel: Union[str, kernel_fn_type],
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior_dist: Optional[dist.Distribution] = None,
                 lengthscale_prior_dist: Optional[dist.Distribution] = None,
                 jitter: float = 1e-6) -> None:

        super(viGP, self).__init__(
            input_dim, kernel, mean_fn, kernel_prior, mean_fn_prior, 
            noise_prior, noise_prior_dist, lengthscale_prior_dist, jitter)

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

    def _print_summary(self) -> None:
        for (k, vals) in self.params.items():
            spaces = " " * (15 - len(k))
            print(k, spaces, jnp.around(vals, 4))
