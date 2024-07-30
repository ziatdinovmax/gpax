from typing import Callable, Dict, Optional

import jax.numpy as jnp
import jax.random as jra
import numpy as onp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

from .mtgp import MultiTaskGP
from ..kernels import LCMKernel
from ..utils import put_on_device


class viMultiTaskGP(MultiTaskGP):
    def __init__(self, input_dim: int, data_kernel: str,
                 num_latents: int = None, shared_input_space: bool = False,
                 num_tasks: int = None, rank: Optional[int] = None,
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 data_kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior_dist: Optional[dist.Distribution] = None,
                 lengthscale_prior_dist: Optional[dist.Distribution] = None,
                 W_prior_dist: Optional[dist.Distribution] = None,
                 v_prior_dist: Optional[dist.Distribution] = None,
                 output_scale: bool = False, jitter: float = 1e-6,
                 **kwargs) -> None:

        super(viMultiTaskGP, self).__init__(input_dim, data_kernel, num_latents, shared_input_space,
                                          num_tasks, rank, mean_fn, data_kernel_prior,
                                          mean_fn_prior, noise_prior, noise_prior_dist,
                                          lengthscale_prior_dist, W_prior_dist, v_prior_dist,
                                          output_scale, jitter, **kwargs)

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

    def _sample_kernel_params(self):
        """
        Sample data ("base") kernel parameters with default weakly-informative
        priors for all the latent functions. Optionally allows to specify a custom
        prior over the kernel lengthscale.
        """
        squeezer = lambda x: x.squeeze() if self.num_latents > 1 else x
        if self.lengthscale_prior_dist is not None:
            length_dist = self.lengthscale_prior_dist
        else:
            length_dist = dist.LogNormal(0.0, 1.0)
        with numpyro.plate("latent_plate_data", self.num_latents, dim=-2):
            with numpyro.plate("ard", self.kernel_dim, dim=-1):
                length = numpyro.sample("k_length", length_dist)
            if self.output_scale:
                scale = numpyro.sample("k_scale", dist.LogNormal(0.0, 1.0))
            else:
                scale = numpyro.sample("k_scale", dist.Normal(1.0, 1e-4))
            if self.data_kernel_name == 'Periodic':
                period = numpyro.sample("period", dist.LogNormal(0.0, 1.0))
        kernel_params = {
            "k_length": squeezer(length), "k_scale": squeezer(scale),
            "period": squeezer(period) if self.data_kernel_name == "Periodic" else None
        }
        return kernel_params

    def get_samples(self, **kwargs):
        return {k: v[None] for (k, v) in self.params.items()}

    def _print_summary(self) -> None:
        for (k, vals) in self.params.items():
            spaces = " " * (15 - len(k))
            print(k, spaces, jnp.around(vals, 4))
