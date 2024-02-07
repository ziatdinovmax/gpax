"""
sparse_gp.py
============

Variational inference implementation of sparse Gaussian process regression

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from typing import Callable, Dict, Optional, Tuple, Type

import jax
import jaxlib
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, solve_triangular

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO

from .vigp import viGP
from ..utils import initialize_inducing_points


class viSparseGP(viGP):
    """
    Variational inference-based sparse Gaussian process

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
        noise_prior:
            Optional custom prior for the observation noise variance; uses LogNormal(0,1) by default.
        guide:
            Auto-guide option, use 'delta' (default) or 'normal'
    """
    def __init__(self, input_dim: int, kernel: str,
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior_dist: Optional[dist.Distribution] = None,
                 lengthscale_prior_dist: Optional[dist.Distribution] = None,
                 guide: str = 'delta') -> None:
        args = (input_dim, kernel, mean_fn, kernel_prior, mean_fn_prior, noise_prior,
                noise_prior_dist, lengthscale_prior_dist, guide)
        super(viSparseGP, self).__init__(*args)
        self.Xu = None

    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              Xu: jnp.ndarray = None,
              **kwargs: float) -> None:
        if Xu is not None:
            Xu = numpyro.param("Xu", Xu)
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[0])
        # Sample kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params()
        # Sample noise
        if self.noise_prior:  # this will be removed in the future releases
            noise = self.noise_prior()
        else:
            noise = self._sample_noise()
        D = jnp.broadcast_to(noise, (X.shape[0],) )
        # Add mean function (if any)
        if self.mean_fn is not None:
            args = [X]
            if self.mean_fn_prior is not None:
                args += [self.mean_fn_prior()]
            f_loc += self.mean_fn(*args).squeeze()
        # Compute kernel between inducing points
        Kuu = self.kernel(Xu, Xu, kernel_params, **kwargs)
        # Cholesky decomposition
        Luu = cholesky(Kuu).T
        # Compute kernel between inducing and training points
        Kuf = self.kernel(Xu, X, kernel_params)
        # Solve triangular system
        W = solve_triangular(Luu, Kuf, lower=True).T
        # Diagonal of the kernel matrix
        Kffdiag = jnp.diag(self.kernel(X, X, kernel_params, jitter=0))
        # Sum of squares computation
        Qffdiag = jnp.square(W).sum(axis=-1)
        # Trace term computation
        trace_term = (Kffdiag - Qffdiag).sum() / noise
        # Clamping the trace term
        trace_term = jnp.clip(trace_term, a_min=0)

        # VFE approximation
        numpyro.factor("trace_term", -trace_term / 2.0)

        numpyro.sample(
            "y",
            dist.LowRankMultivariateNormal(loc=f_loc, cov_factor=W, cov_diag=D),
            obs=y)

    def fit(self,
            rng_key: jnp.array, X: jnp.ndarray, y: jnp.ndarray,
            inducing_points_ratio: float = 0.1, inducing_points_selection: str = 'uniform',
            num_steps: int = 1000, step_size: float = 5e-3,
            progress_bar: bool = True, print_summary: bool = True,
            device: Type[jaxlib.xla_extension.Device] = None,
            **kwargs: float
            ) -> None:
        """
        Run variational inference to learn GP (hyper)parameters

        Args:
            rng_key: random number generator key
            X: 2D feature vector with *(number of points, number of features)* dimensions
            y: 1D target vector with *(n,)* dimensions
            Xu: Inducing points ratio. Must be a float between 0 and 1. Default value is 0.1.
            num_steps: number of SVI steps
            step_size: step size schedule for Adam optimizer
            progress_bar: show progress bar
            print_summary: print summary at the end of training
            device:
                optionally specify a cpu or gpu device on which to run the inference;
                e.g., ``device=jax.devices("cpu")[0]``
            **jitter:
                Small positive term added to the diagonal part of a covariance
                matrix for numerical stability (Default: 1e-6)
        """
        X, y = self._set_data(X, y)
        if device:
            X = jax.device_put(X, device)
            y = jax.device_put(y, device)
        Xu = initialize_inducing_points(
            X.copy(), inducing_points_ratio,
            inducing_points_selection, rng_key)
        self.X_train = X
        self.y_train = y

        optim = numpyro.optim.Adam(step_size=step_size, b1=0.5)
        self.svi = SVI(
            self.model,
            guide=self.guide_type(self.model),
            optim=optim,
            loss=Trace_ELBO(),
            X=X,
            y=y,
            Xu=Xu,
            **kwargs
        )

        self.kernel_params = self.svi.run(
            rng_key, num_steps, progress_bar=progress_bar)[0]

        self.Xu = self.kernel_params['Xu']

        if print_summary:
            self._print_summary()

    def get_mvn_posterior(
        self, X_new: jnp.ndarray, params: Dict[str, jnp.ndarray], noiseless: bool = False, **kwargs: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns parameters (mean and cov) of multivariate normal posterior
        for a single sample of GP parameters
        """
        noise = params["noise"]
        N = self.X_train.shape[0]
        D = jnp.broadcast_to(noise, (N,))
        noise_p = noise * (1 - jnp.array(noiseless, int))

        y_residual = self.y_train.copy()
        if self.mean_fn is not None:
            args = [self.X_train, params] if self.mean_fn_prior else [self.X_train]
            y_residual -= self.mean_fn(*args).squeeze()

        # Compute self- and cross-covariance matrices
        Kuu = self.kernel(self.Xu, self.Xu, params, **kwargs)
        Luu = cholesky(Kuu, lower=True)
        Kuf = self.kernel(self.Xu, self.X_train, params, jitter=0)

        W = solve_triangular(Luu, Kuf, lower=True)
        W_Dinv = W / D
        K = W_Dinv @ W.T
        K = K.at[jnp.diag_indices(K.shape[0])].add(1)
        L = cholesky(K, lower=True)

        y_2D = y_residual.reshape(-1, N).T
        W_Dinv_y = W_Dinv @ y_2D

        Kus = self.kernel(self.Xu, X_new, params, jitter=0)
        Ws = solve_triangular(Luu, Kus, lower=True)
        pack = jnp.concatenate((W_Dinv_y, Ws), axis=1)
        Linv_pack = solve_triangular(L, pack, lower=True)

        Linv_W_Dinv_y = Linv_pack[:, :W_Dinv_y.shape[1]]
        Linv_Ws = Linv_pack[:, W_Dinv_y.shape[1]:]
        mean = (Linv_W_Dinv_y.T @ Linv_Ws).squeeze()

        Kss = self.kernel(X_new, X_new, params, noise_p, **kwargs)
        Qss = Ws.T @ Ws
        cov = Kss - Qss + Linv_Ws.T @ Linv_Ws

        if self.mean_fn is not None:
            args = [X_new, params] if self.mean_fn_prior else [X_new]
            mean += self.mean_fn(*args).squeeze()

        return mean, cov