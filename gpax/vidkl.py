from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.contrib.module import haiku_module
from jax import jit
import haiku as hk

from .gp import ExactGP
from .kernels import get_kernel
from .utils import get_keys


class viDKL(ExactGP):
    """
    Implementation of deep kernel learning inspired by arXiv:1511.02222

    Args:
        input_dim: number of input dimensions
        z_dim: latent space dimensionality
        kernel: type of kernel ('RBF', 'Matern', 'Periodic')
        kernel_prior: optional priors over kernel hyperparameters (uses LogNormal(0,1) by default)
        nn: Custom neural network (optional)
        latent_prior: Optional prior over the latent space (NN embedding)
    """

    def __init__(self, input_dim: int, z_dim: int = 2, kernel: str = 'RBF',
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 nn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
                 latent_prior: Optional[Callable[[jnp.ndarray], Dict[str, jnp.ndarray]]] = None
                 ) -> None:
        super(viDKL, self).__init__(input_dim, kernel, kernel_prior)
        nn_module = nn if nn else MLP
        self.nn_module = hk.transform(lambda x: nn_module(z_dim)(x))
        self.kernel_dim = z_dim
        self.data_dim = (input_dim,) if isinstance(input_dim, int) else input_dim
        self.latent_prior = latent_prior
        self.kernel_params = None
        self.nn_params = None
        self.ensemble_mode = False

    def model(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """DKL probabilistic model"""
        task_dim = X.shape[0]
        # NN part
        feature_nets = [
            haiku_module("feature_net_{}".format(i), self.nn_module, input_shape=(1, *self.data_dim))
            for i in range(task_dim)
        ]
        z = jnp.array([f(X[i]) for (i, f) in enumerate(feature_nets)])
        # Sample GP kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params(task_dim)
        # Sample noise
        with numpyro.plate('obs_noise', task_dim):
            noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
        # GP's mean function
        f_loc = jnp.zeros(z.shape[:2])
        # compute kernel(s)
        k_args = (z, z, kernel_params, noise)
        k = jax.jit(jax.vmap(get_kernel(self.kernel)))(*k_args)
        # sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def fit(self, rng_key: jnp.array, X: jnp.ndarray, y: jnp.ndarray,
            num_steps: int = 1000, step_size: float = 5e-3,
            print_summary: bool = True) -> None:
        """
        Run SVI to infer the GP model parameters

        Args:
            rng_key: random number generator key
            X: 2D 'feature vector' with :math:`n x num_features` dimensions
            y: 1D 'target vector' with :math:`(n,)` dimensions
            num_steps: number of SVI steps
            step_size: step size schedule for Adam optimizer
            print_summary: print summary at the end of sampling
        """
        X, y = self._set_data(X, y)
        self.X_train = X
        self.y_train = y
        # Setup optimizer and SVI
        optim = numpyro.optim.Adam(step_size=step_size, b1=0.5)
        svi = SVI(
            self.model,
            guide=AutoDelta(self.model),
            optim=optim,
            loss=Trace_ELBO(),
            X=X,
            y=y,
        )
        params = svi.run(rng_key, num_steps)[0]
        # Get NN weights
        self.nn_params = [
            params["feature_net_{}$params".format(i)] for i in range(X.shape[0])
        ]
        # Get kernel parameters from the guide
        self.kernel_params = svi.guide.median(params)
        if print_summary:
            self._print_summary()

    def fit_ensemble(self, rng_key: jnp.array,
                     X: jnp.ndarray, y: jnp.ndarray,
                     n_models: int, num_steps: int = 1000,
                     step_size: float = 5e-3, print_summary: bool = True
                     ) -> None:
        self.ensemble_mode = True
        X, y = self._set_data(X, y)
        X = X.repeat(n_models, axis=0)
        y = y.repeat(n_models, axis=0)
        self.fit(rng_key, X, y, num_steps, step_size, print_summary)

    def _get_mvn_posterior(self,
                           z_train: jnp.ndarray,
                           y_train: jnp.ndarray,
                           z_test: jnp.ndarray,
                           params: Dict[str, jnp.ndarray] = None
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        noise = params["noise"]
        # compute kernel matrices for train and test data
        k_pp = get_kernel(self.kernel)(z_test, z_test, params, noise)
        k_pX = get_kernel(self.kernel)(z_test, z_train, params, jitter=0.0)
        k_XX = get_kernel(self.kernel)(z_train, z_train, params, noise)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_train))
        return mean, cov

    @partial(jit, static_argnames='self')
    def get_mvn_posterior(self,
                          X_new: jnp.ndarray,
                          params: Dict[str, jnp.ndarray] = None
                          ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns predictive mean and covariance at new points
        (mean and cov, where cov.diagonal() is 'uncertainty')
        given a single set of DKL hyperparameters
        """
        if params is None:
            params = self.kernel_params
        z_train = self.embed(self.X_train)
        z_new = self.embed(X_new)
        vmap_args = (z_train, self.y_train, z_new, params)
        mean, cov = jax.vmap(self._get_mvn_posterior)(*vmap_args)
        return mean, cov

    def predict(self, rng_key: jnp.ndarray, X_new: jnp.ndarray,
                kernel_params: Optional[Dict[str, jnp.ndarray]] = None,
                n: int = 5000, *args
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

    @partial(jit, static_argnames='self')
    def embed(self, X_new: jnp.ndarray) -> jnp.ndarray:
        """
        Embeds data into the latent space using
        the DKL's (trained) neural network
        """
        task_dim = self.X_train.shape[0]
        X_new = self._set_data(X_new)
        if X_new.shape[0] != task_dim and self.ensemble_mode:
            X_new = X_new.repeat(task_dim, axis=0)
        else:
            raise AssertionError(
                "The batch/task dimension of new data must match" +
                " the batch/task dimension of train data")
        key, _ = get_keys()
        z = [
            self.nn_module.apply(self.nn_params[i], key, X_new[i])
            for i in range(task_dim)
        ]
        return jnp.array(z)

    def _print_summary(self) -> None:
        if isinstance(self.kernel_params, dict):
            print('\nInferred parameters')
            if self.X_train.shape[0] == 1:
                for (k, vals) in self.kernel_params.items():
                    spaces = " " * (15 - len(k))
                    print(k, spaces, jnp.around(vals, 4))
            else:
                for (k, vals) in self.kernel_params.items():
                    for i, v in enumerate(vals):
                        spaces = " " * (15 - len(k))
                        print(k+"[{}]".format(i), spaces, jnp.around(v, 4))


class MLP(hk.Module):
    def __init__(self, embedim=2):
        super().__init__()
        self._embedim = embedim

    def __call__(self, x):
        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._embedim)(x)
        return x
