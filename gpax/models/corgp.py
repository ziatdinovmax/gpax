from typing import Callable, Dict, Optional

import jax.numpy as jnp
import numpy as onp
import numpyro
import numpyro.distributions as dist

from .gp import ExactGP
from ..kernels import MultitaskKernel


class CoregGP(ExactGP):

    """
    Coregionalized Gaussian Process model

    Args:
        input_dim:
            Number of input dimensions
        data_kernel:
            Kernel function operating on data inputs ('RBF', 'Matern', 'Periodic', or a custom function)
        mean_fn:
            Optional deterministic mean function (use 'mean_fn_priors' to make it probabilistic)
        data_kernel_prior:
            Optional custom priors over the data kernel hyperparameters; uses LogNormal(0,1) by default
        mean_fn_prior:
            Optional priors over mean function parameters
        noise_prior:
            Optional custom prior for observation noise variance; uses LogNormal(0,1) by default.
        task_kernel_prior:
            Optional custom priors over task kernel parameters;
            Defaults to Normal(0, 10) for weights W and LogNormal(0, 1) for variances v.
        rank: int
            Rank of the weight matrix in the task kernel. Cannot be larger than the number of tasks.
            Higher rank implies higher correlation. Defaults to 1.

    """
    def __init__(self, input_dim: int, data_kernel: str,
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 data_kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 task_kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 rank: int = 1, **kwargs) -> None:
        args = (input_dim, None, mean_fn, None, mean_fn_prior, noise_prior)
        super(CoregGP, self).__init__(*args)
        self.num_tasks = None
        self.rank = rank
        self.kernel = MultitaskKernel(data_kernel, **kwargs)
        self.data_kernel_prior = data_kernel_prior
        self.task_kernel_prior = task_kernel_prior
        self.kernel_name = data_kernel

    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              **kwargs: float
              ) -> None:
        """Multitask GP probabilistic model with inputs X and targets y"""
        self.num_tasks = len(onp.unique(X[:, -1]))
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[0])

        # Sample data kernel parameters
        if self.data_kernel_prior:
            data_kernel_params = self.data_kernel_prior()
        else:
            data_kernel_params = self._sample_kernel_params(output_scale=False)

        # Sample task kernel parameters
        if self.task_kernel_prior:
            task_kernel_params = self.task_kernel_prior()
        else:
            task_kernel_params = self._sample_task_kernel_params(self.num_tasks, self.rank)

        # Combine two dictionaries with parameters
        kernel_params = {**data_kernel_params, **task_kernel_params}

        # Sample noise
        if self.noise_prior:
            noise = self.noise_prior()
        else:  # consider using numpyro.plate here
            noise = numpyro.sample(
                "noise", dist.LogNormal(
                    jnp.zeros(self.num_tasks), jnp.ones(self.num_tasks))
            )

        # Compute multitask_kernel
        k = self.kernel(X, X, kernel_params, noise)

        # Add mean function (if any)
        if self.mean_fn is not None:
            args = [X]
            if self.mean_fn_prior is not None:
                args += [self.mean_fn_prior()]
            f_loc += self.mean_fn(*args).squeeze()

        # sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def _sample_task_kernel_params(self, n_tasks, rank):
        """
        Sample task kernel parameters with default weakly-informative priors
        """
        W = numpyro.sample("W", numpyro.distributions.Normal(
            jnp.zeros(shape=(n_tasks, rank)), 10*jnp.ones(shape=(n_tasks, rank))))
        v = numpyro.sample("v", numpyro.distributions.LogNormal(
            jnp.zeros(shape=(n_tasks,)), jnp.ones(shape=(n_tasks,))))
        return {"W": W, "v": v}
