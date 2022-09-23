"""
vgp.py
=======

Fully Bayesian implementation of Gaussian process regression for vector-valued functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union, Type

import jax
import jaxlib
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import jit

from .gp import ExactGP
from .kernels import get_kernel


class vExactGP(ExactGP):
    """
    Gaussian process class for vector-valued targets

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
        args = (input_dim, kernel, mean_fn,  kernel_prior, mean_fn_prior, noise_prior)
        super(vExactGP, self).__init__(*args)

    def model(self, 
              X: jnp.ndarray,
              y: jnp.ndarray = None, 
              **kwargs: float
              ) -> None:
        """GP probabilistic model with inputs X and vector-valued targets y"""
        jitter = kwargs.get("jitter", 1e-6)
        task_dim = X.shape[0]
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[:2])
        # Sample parameters of kernels
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params(task_dim=task_dim)
        # Sample noise for each task
        with numpyro.plate("noise_plate", task_dim):
            if self.noise_prior:
                noise = self.noise_prior()
            else:
                noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
        # Add mean function (if any)
        if self.mean_fn is not None:
            args = [X]
            if self.mean_fn_prior is not None:
                args += [self.mean_fn_prior()]
            f_loc += self.mean_fn(*args).squeeze()
        # Compute kernels for each task in parallel
        jitter = jnp.array(jitter).repeat(task_dim)
        k_args = (X, X, kernel_params, noise)
        k = jax.vmap(get_kernel(self.kernel))(*k_args, jitter=jitter)
        # Sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    @partial(jit, static_argnames='self')
    def _get_mvn_posterior(self,
                           X_train: jnp.ndarray, y_train: jnp.ndarray,
                           X_new: jnp.ndarray, params: Dict[str, jnp.ndarray],
                           m_X: Optional[jnp.ndarray] = None,
                           m_p: Optional[jnp.ndarray] = None,
                           noiseless: bool = False, **kwargs: float
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        noise = params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        y_residual = y_train
        if m_X is not None:
            y_residual -= m_X
        # compute kernel matrices for train and test data
        k_pp = get_kernel(self.kernel)(X_new, X_new, params, noise_p, **kwargs)
        k_pX = get_kernel(self.kernel)(X_new, X_train, params, jitter=0.0)
        k_XX = get_kernel(self.kernel)(X_train, X_train, params, noise, **kwargs)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))
        if m_p is not None:
            mean += m_p
        return mean, cov

    def get_mvn_posterior(self,
                          X_new: jnp.ndarray, params: Dict[str, jnp.ndarray],
                          noiseless: bool = False, **kwargs: float
                          ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns parameters (mean and cov) of multivariate normal posterior
        for a single sample of GP parameters. Wrapper over self._get_mvn_posterior.
        """
        task_dim = X_new.shape[0]
        jitter = kwargs.get("jitter", 1e-6)
        if self.mean_fn is not None:  # Compute mean function for training data and new data
            get_args = lambda x: [x, params] if self.mean_fn_prior else [x]
            m_X = self.mean_fn(*get_args(self.X_train)).squeeze()
            m_p = self.mean_fn(*get_args(X_new)).squeeze()
            params_unsqueezed = {   # ensure all params has the same 'batch dimension' for vmap to work
                k: p[None].repeat(X_new.shape[0], axis=0) if p.ndim == 0 else p
                for k, p in params.items()
            }
            vmap_args = (
                self.X_train, self.y_train, X_new, params_unsqueezed, m_X, m_p)
        else:
            vmap_args = (self.X_train, self.y_train, X_new, params)
        noiseless = jnp.array(noiseless, int).repeat(task_dim)
        jitter = jnp.array(jitter).repeat(task_dim)
        mean, cov = jax.vmap(
            self._get_mvn_posterior)(*vmap_args, noiseless=noiseless, jitter=jitter)
        return mean, cov

    def _sample_kernel_params(self, task_dim: int = None) -> Dict[str, jnp.ndarray]:
        """
        Sample kernel parameters with default
        weakly-informative log-normal priors
        """
        with numpyro.plate("plate_1", task_dim, dim=-2):  # task dimension
            with numpyro.plate('lengthscale', self.kernel_dim, dim=-1):  # allows using ARD kernel for kernel_dim > 1
                length = numpyro.sample("k_length", dist.LogNormal(0.0, 1.0))
        with numpyro.plate("plate_2", task_dim):  # task dimension'
            scale = numpyro.sample("k_scale", dist.LogNormal(0.0, 1.0))
            if self.kernel == 'Periodic':
                period = numpyro.sample("period", dist.LogNormal(0.0, 1.0))
        kernel_params = {
            "k_length": length, "k_scale": scale,
            "period": period if self.kernel == "Periodic" else None}
        return kernel_params

    def predict_in_batches(self, rng_key: jnp.ndarray,
                           X_new: jnp.ndarray,  batch_size: int = 100,
                           samples: Optional[Dict[str, jnp.ndarray]] = None,
                           n: int = 1, filter_nans: bool = False,
                           predict_fn: Callable[[jnp.ndarray, int], Tuple[jnp.ndarray]] = None,
                           noiseless: bool = False,
                           device: Type[jaxlib.xla_extension.Device] = None,
                           **kwargs: float
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new with sampled GP parameters
        by spitting the input array into chunks ("batches") and running
        predict_fn (defaults to self.predict) on each of them one-by-one
        to avoid a memory overflow
        """
        X_new = self._set_data(X_new)
        y_pred, y_sampled = self._predict_in_batches(
            rng_key, X_new, batch_size, 1, samples, n,
            filter_nans, predict_fn, noiseless, device, **kwargs)
        y_pred = jnp.concatenate(y_pred, -1)
        y_sampled = jnp.concatenate(y_sampled, -1)
        return y_pred, y_sampled

    def _set_data(self,
                  X: jnp.ndarray,
                  y: Optional[jnp.ndarray] = None
                  ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X[..., None] if X.ndim == 2 else X  # add feature pseudo-dimension
        if y is not None:
            if y.shape[0] != X.shape[0]:
                raise AssertionError(
                    "Task dimensions must be identical in inputs and targets")
            return X, y
        return X
