from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
from jax import jit
import jax.numpy as jnp
import numpy as onp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_haiku_module

from ..kernels import LCMKernel
from .vidkl import viDKL


class viMTDKL(viDKL):
    """
    Implementation of the variational infernece-based deep kernel learning
    for multi-task/fidelity problems

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
                 nn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
                 guide: str = 'delta',
                 W_prior_dist: Optional[dist.Distribution] = None,
                 v_prior_dist: Optional[dist.Distribution] = None,
                 task_kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 **kwargs) -> None:
        args = (input_dim, z_dim, None, None, nn, None, guide)
        super(viMTDKL, self).__init__(*args, **kwargs)
        if shared_input_space:
            if num_tasks is None:
                raise ValueError("Please specify num_tasks")
        else:
            if num_latents is None:
                raise ValueError("Please specify num_latents")
        self.num_tasks = num_tasks
        self.num_latents = num_tasks if num_latents is None else num_latents
        self.rank = rank
        self.kernel = LCMKernel(
            data_kernel, shared_input_space, num_tasks, **kwargs)
        self.data_kernel_prior = data_kernel_prior
        self.task_kernel_prior = task_kernel_prior
        self.shared_input = shared_input_space
        self.W_prior_dist = W_prior_dist
        self.v_prior_dist = v_prior_dist

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """Multitask DKL probabilistic model"""

        # Check that we have necessary info for sampling kernel params
        if not self.shared_input and self.num_tasks is None:
            self.num_tasks = len(onp.unique(self.X_train[:, -1]))

        if self.rank is None:
            self.rank = self.num_tasks - 1

        # NN part
        feature_extractor = random_haiku_module(
            "feature_extractor", self.nn_module, input_shape=(1, *self.data_dim),
            prior=(lambda name, shape: dist.Cauchy() if name.startswith("b") else dist.Normal()))
        z = feature_extractor(X if self.shared_input else X[:, :-1])
        if not self.shared_input:
            z = jnp.column_stack((z, X[:, -1]))

        # Initialize GP kernel mean function at zeros
        if self.shared_input:
            f_loc = jnp.zeros(self.num_tasks * X.shape[0])
        else:
            f_loc = jnp.zeros(X.shape[0])

        # Sample data kernel parameters
        if self.data_kernel_prior:
            data_kernel_params = self.data_kernel_prior()
        else:
            data_kernel_params = self._sample_kernel_params()

        # Sample task kernel parameters
        if self.task_kernel_prior:
            task_kernel_params = self.task_kernel_prior()
        else:
            task_kernel_params = self._sample_task_kernel_params()

        # Combine two dictionaries with parameters
        kernel_params = {**data_kernel_params, **task_kernel_params}

        # Sample noise
        if self.noise_prior:  # this will be removed in the future releases
            noise = self.noise_prior()
        else:
            noise = self._sample_noise()

        # Compute multitask_kernel
        k = self.kernel(z, z, kernel_params, noise, **kwargs)

        # Sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def _sample_noise(self):
        """Sample observational noise"""
        if self.noise_prior_dist is not None:
            noise_dist = self.noise_prior_dist
        else:
            noise_dist = dist.LogNormal(
                    jnp.zeros(self.num_tasks),
                    jnp.ones(self.num_tasks))

        noise = numpyro.sample("noise", noise_dist.to_event(1))
        return noise

    def _sample_task_kernel_params(self):
        """
        Sample task kernel parameters with default weakly-informative priors
        or custom priors for all the latent functions
        """
        if self.W_prior_dist is not None:
            W_dist = self.W_prior_dist
        else:
            W_dist = dist.Normal(
                    jnp.zeros(shape=(self.num_latents, self.num_tasks, self.rank)),  # loc
                    10*jnp.ones(shape=(self.num_latents, self.num_tasks, self.rank)) # var
            )
        if self.v_prior_dist is not None:
            v_dist = self.v_prior_dist
        else:
            v_dist = dist.LogNormal(
                    jnp.zeros(shape=(self.num_latents, self.num_tasks)),  # loc
                    jnp.ones(shape=(self.num_latents, self.num_tasks)) # var
            )
        with numpyro.plate("latent_plate_task", self.num_latents):
            W = numpyro.sample("W", W_dist.to_event(2))
            v = numpyro.sample("v", v_dist.to_event(1))
        return {"W": W, "v": v}

    def _sample_kernel_params(self):
        """
        Sample data ("base") kernel parameters with default weakly-informative
        priors for all the latent functions
        """
        squeezer = lambda x: x.squeeze() if self.num_latents > 1 else x
        with numpyro.plate("latent_plate_data", self.num_latents, dim=-2):
            with numpyro.plate("ard", self.kernel_dim, dim=-1):
                length = numpyro.sample("k_length", dist.LogNormal(0.0, 1.0))
            scale = numpyro.sample("k_scale", dist.Normal(1.0, 1e-4))
        return {"k_length": squeezer(length), "k_scale": squeezer(scale)}

    #@partial(jit, static_argnames='self')
    def get_mvn_posterior(self,
                          X_new: jnp.ndarray,
                          nn_params: Dict[str, jnp.ndarray],
                          k_params: Dict[str, jnp.ndarray],
                          noiseless: bool = False,
                          y_residual: jnp.ndarray = None,
                          **kwargs
                          ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns predictive mean and covariance at new points
        (mean and cov, where cov.diagonal() is 'uncertainty')
        given a single set of DKL parameters
        """
        if y_residual is None:
            y_residual = self.y_train
        noise = k_params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        # embed data into the latent space
        z_train = self.nn_module.apply(
            nn_params, jax.random.PRNGKey(0),
            self.X_train if self.shared_input else self.X_train[:, :-1])
        z_test = self.nn_module.apply(
            nn_params, jax.random.PRNGKey(0),
            X_new if self.shared_input else X_new[:, :-1])
        if not self.shared_input:
            z_train = jnp.column_stack((z_train, self.X_train[:, -1]))
            z_test = jnp.column_stack((z_test, X_new[:, -1]))
        # compute kernel matrices for train and test data
        k_pp = self.kernel(z_test, z_test, k_params, noise_p, **kwargs)
        k_pX = self.kernel(z_test, z_train, k_params, jitter=0.0)
        k_XX = self.kernel(z_train, z_train, k_params, noise, **kwargs)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))
        return mean, cov
