"""
vi_mtdkl.py
========

Variational inference-based implementation of multi-task deep kernel learning

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from functools import partial
from typing import Callable, Dict, Optional, Tuple, Type, List

import jax.random as jra
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import haiku as hk

from .mtdkl import MultiTaskDKL
from ..utils import put_on_device, get_haiku_compatible_dict


class viMultiTaskDKL(MultiTaskDKL):
    """
    Variational inference-based deep kernel learning for multi-task/fidelity problems

    Args:
        input_dim:
            Number of input dimensions, not counting the column with task indices (if any)
        z_dim:
            Latent space dimensionality (defaults to 2)
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
        data_kernel_prior:
            Optional priors over kernel hyperparameters; uses LogNormal(0,1) by default
        nn:
            Custom neural network ('feature extractor'); uses a 3-layer MLP
            with ReLU activations by default
        nn_prior:
            Places probabilistic priors over NN weights and biases (Default: True)
        latent_prior:
            Optional prior over the latent space (NN embedding); uses none by default
        guide:
            Auto-guide option, use 'delta' (default) or 'normal'
        W_prior_dist:
            Optional custom prior distribution over W in the task kernel, :math:`WW^T + diag(v)`.
            Defaults to Normal(0, 10).
        v_prior_dist:
            Optional custom prior distribution over v in the task kernel, :math:`WW^T + diag(v)`.
            Must be non-negative. Defaults to LogNormal(0, 1)
        task_kernel_prior:
            Optional custom priors over task kernel parameters;
            Defaults to Normal(0, 10) for weights W and LogNormal(0, 1) for variances v.

        **kwargs:
            Optional custom prior distributions over observational noise (noise_dist_prior)
            and kernel lengthscale (lengthscale_prior_dist)
    """

    def __init__(self, input_dim: int, z_dim: int = 2, data_kernel: str = 'RBF',
                 num_latents: int = None, shared_input_space: bool = False,
                 num_tasks: int = None, rank: Optional[int] = None,
                 data_kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 hidden_dim: Optional[List[int]] = None, activation: str = 'relu',
                 nn: Type[hk.Module] = None,
                 W_prior_dist: Optional[dist.Distribution] = None,
                 v_prior_dist: Optional[dist.Distribution] = None,
                 task_kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 jitter: float = 1e-6,
                 **kwargs) -> None:
        super(viMultiTaskDKL, self).__init__(input_dim, z_dim, data_kernel, num_latents,
                                             shared_input_space, num_tasks, rank, data_kernel_prior,
                                             hidden_dim, activation, nn, W_prior_dist, v_prior_dist, task_kernel_prior, jitter)
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_steps: int = 1000, step_size: float = 5e-3,
            progress_bar: bool = True,
            print_summary: bool = True,
            device: str = None,
            rng_key: jnp.array = None,
            **kwargs: float
            ) -> None:
        """
        Run variational inference to learn DKL (hyper)parameters

        Args:
            rng_key: random number generator key
            X: 2D feature vector with *(number of points, number of features)* dimensions
            y: 1D target vector with *(n,)* dimensions
            num_steps: number of SVI steps
            step_size: step size schedule for Adam optimizer
            progress_bar: show progress bar
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
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

        if print_summary:
            self.print_summary()

    def get_samples(self, **kwargs):
        return get_haiku_compatible_dict(self.params, map=True)  # map=True adds an extra batch dimension to make it work with vmap

    def print_summary(self, print_nn_weights: bool = False) -> None:
        list_of_keys = ["k_scale", "k_length", "noise"]
        print('\nInferred GP kernel parameters')
        for (k, vals) in self.params.items():
            if k in list_of_keys:
                spaces = " " * (15 - len(k))
                print(k, spaces, jnp.around(vals, 4))

    def _sample_scale(self):
        return numpyro.sample("k_scale", dist.Normal(1.0, 1e-4))
