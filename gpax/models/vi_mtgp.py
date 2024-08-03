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
    """
    Variational Inference Gaussian process for multi-task/fidelity learning

    Args:
        input_dim:
            Number of input dimensions
        data_kernel:
            Kernel function operating on data inputs ('RBF', 'Matern', 'Periodic', or a custom function)
        num_latents:
            Number of latent functions. Typically equal to or less than the number of tasks
        shared_input_space:
            If True, assumes that all tasks share the same input space and
            uses a multivariate kernel (Kronecker product). If False (default), assumes that different tasks
            have different number of observations and uses a multitask kernel (elementwise multiplication).
            In that case, the task indices must be appended as the last column of the input vector.
        num_tasks:
            Number of tasks. This is only needed if `shared_input_space` is True.
        rank:
            Rank of the weight matrix in the task kernel. Cannot be larger than the number of tasks.
            Higher rank implies higher correlation. Uses *(num_tasks - 1)* when not specified.
        mean_fn:
            Optional deterministic mean function (use 'mean_fn_priors' to make it probabilistic)
        data_kernel_prior:
            Optional custom priors over the data kernel hyperparameters
        mean_fn_prior:
            Optional priors over mean function parameters
        noise_prior_dist:
            Optional custom prior distribution over the observational noise variance.
            Defaults to LogNormal(0,1).
        lengthscale_prior_dist:
            Optional custom prior distribution over kernel lengthscale. Defaults to LogNormal(0, 1)
        W_prior_dist:
            Optional custom prior distribution over W in the task kernel, :math:`WW^T + diag(v)`.
            Defaults to Normal(0, 10).
        v_prior_dist:
            Optional custom prior distribution over v in the task kernel, :math:`WW^T + diag(v)`.
            Must be non-negative. Defaults to LogNormal(0, 1)
        task_kernel_prior:
            Optional custom priors over task kernel parameters;
            Defaults to Normal(0, 10) for weights W and LogNormal(0, 1) for variances v.
        output_scale:
            Option to sample data kernel's output scale.
            Defaults to False to avoid over-parameterization (the scale is already absorbed into task kernel).
        jitter:
            Small jitter for the numerical stability. Default: 1e-6.
    """
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
            print_summary: bool = True,
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

        if print_summary:
            self.print_summary()

    def _sample_scale(self):
        if self.output_scale:
            return numpyro.sample("k_scale", dist.LogNormal(0.0, 1.0))
        else:
            return numpyro.sample("k_scale", dist.Normal(1.0, 1e-4))

    def get_samples(self, **kwargs):
        return {k: v[None] for (k, v) in self.params.items()}

    def print_summary(self) -> None:
        for (k, vals) in self.params.items():
            spaces = " " * (15 - len(k))
            print(k, spaces, jnp.around(vals, 4))
