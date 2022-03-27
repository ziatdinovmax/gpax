from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal
from numpyro.contrib.module import random_haiku_module
from jax import jit
import haiku as hk

from .gp import ExactGP
from .kernels import get_kernel
from .utils import get_haiku_dict


class viDKL(ExactGP):
    """
    Implementation of the variational infernece-based deep kernel learning

    Args:
        input_dim: number of input dimensions
        z_dim: latent space dimensionality
        kernel: type of kernel ('RBF', 'Matern', 'Periodic')
        kernel_prior: optional priors over kernel hyperparameters (uses LogNormal(0,1) by default)
        nn: Custom neural network (optional)
        latent_prior: Optional prior over the latent space (NN embedding)
        guide: auto-guide option, use 'delta' (default) or 'normal'
    """

    def __init__(self, input_dim: int, z_dim: int = 2, kernel: str = 'RBF',
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 nn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
                 latent_prior: Optional[Callable[[jnp.ndarray], Dict[str, jnp.ndarray]]] = None,
                 guide: str = 'delta'
                 ) -> None:
        super(viDKL, self).__init__(input_dim, kernel, kernel_prior)
        if guide not in ['delta', 'normal']:
            raise NotImplementedError("Select guide between 'delta' and 'normal'")
        nn_module = nn if nn else MLP
        self.nn_module = hk.transform(lambda x: nn_module(z_dim)(x))
        self.kernel_dim = z_dim
        self.data_dim = (input_dim,) if isinstance(input_dim, int) else input_dim
        self.latent_prior = latent_prior
        self.guide_type = AutoNormal if guide == 'normal' else AutoDelta
        self.kernel_params = None
        self.nn_params = None

    def model(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """DKL probabilistic model"""
        # NN part
        feature_extractor = random_haiku_module(
            "feature_extractor", self.nn_module, input_shape=(1, *self.data_dim),
            prior=(lambda name, shape: dist.Cauchy() if name.startswith("b") else dist.Normal()))
        z = feature_extractor(X)
        if self.latent_prior:  # Sample latent variable
            z = self.latent_prior(z)
        # Sample GP kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params()
        # Sample noise
        noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
        # GP's mean function
        f_loc = jnp.zeros(z.shape[0])
        # compute kernel
        k = get_kernel(self.kernel)(
            z, z,
            kernel_params,
            noise
        )
        # sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def single_fit(self, rng_key: jnp.array, X: jnp.ndarray, y: jnp.ndarray,
                   num_steps: int = 1000, step_size: float = 5e-3,
                   print_summary: bool = True, progress_bar=True) -> None:
        """
        Optimizes parameters of a single DKL model
        """
        # Setup optimizer and SVI
        optim = numpyro.optim.Adam(step_size=step_size, b1=0.5)
        svi = SVI(
            self.model,
            guide=self.guide_type(self.model),
            optim=optim,
            loss=Trace_ELBO(),
            X=X,
            y=y,
        )
        params, _, losses = svi.run(rng_key, num_steps, progress_bar=progress_bar)
        # Get DKL parameters from the guide
        params_map = svi.guide.median(params)
        # Get NN weights
        nn_params = get_haiku_dict(params_map)
        # Get GP kernel hyperparmeters
        kernel_params = {k: v for (k, v) in params_map.items()
                         if not k.startswith("feature_extractor")}
        return nn_params, kernel_params, losses

    def fit(self, rng_key: jnp.array, X: jnp.ndarray, y: jnp.ndarray,
            num_steps: int = 1000, step_size: float = 5e-3,
            print_summary: bool = True, progress_bar=True):
        """
        Run stochastic variational inference to learn a DKL model(s) parameters

        Args:
            rng_key: random number generator key
            X: Input high-dimensional features
            y: Target output (scalar of vector)
            num_steps: number of SVI steps
            step_size: step size schedule for Adam optimizer
            print_summary: print summary at the end of sampling
            progress_bar: show progress bar (works only for scalar outputs)
        """
        def _single_fit(x_i, y_i):
            return self.single_fit(
                rng_key, x_i, y_i, num_steps, step_size,
                print_summary=False, progress_bar=False)

        self.X_train = X
        self.y_train = y

        if X.ndim == len(self.data_dim) + 2:
            self.nn_params, self.kernel_params, self.loss = jax.vmap(_single_fit)(X, y)
            if progress_bar:
                avg_bw = [num_steps - num_steps // 20, num_steps]
                print("init loss: {}, final loss (avg) [{}-{}]: {} ".format(
                    self.loss[0].mean(), avg_bw[0], avg_bw[1],
                    self.loss.mean(0)[avg_bw[0]:avg_bw[1]].mean().round(4)))
        else:
            self.nn_params, self.kernel_params, self.loss = self.single_fit(
                rng_key, X, y, num_steps, step_size, print_summary, progress_bar
            )
        if print_summary:
            self._print_summary()

    @partial(jit, static_argnames='self')
    def get_mvn_posterior(self,
                          X_train: jnp.ndarray,
                          y_train: jnp.ndarray,
                          X_new: jnp.ndarray,
                          nn_params: Dict[str, jnp.ndarray],
                          k_params: Dict[str, jnp.ndarray],
                          noiseless: bool = False
                          ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns predictive mean and covariance at new points
        (mean and cov, where cov.diagonal() is 'uncertainty')
        given a single set of DKL hyperparameters
        """
        noise = k_params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        # embed data into the latent space
        z_train = self.nn_module.apply(
            nn_params, jax.random.PRNGKey(0), X_train)
        z_test = self.nn_module.apply(
            nn_params, jax.random.PRNGKey(0), X_new)
        # compute kernel matrices for train and test data
        k_pp = get_kernel(self.kernel)(z_test, z_test, k_params, noise_p)
        k_pX = get_kernel(self.kernel)(z_test, z_train, k_params, jitter=0.0)
        k_XX = get_kernel(self.kernel)(z_train, z_train, k_params, noise)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_train))
        return mean, cov

    def sample_from_posterior(self, rng_key: jnp.ndarray,
                              X_new: jnp.ndarray, n: int = 1000,
                              noiseless: bool = False
                              ) -> Tuple[jnp.ndarray]:
        """
        Samples from the DKL posterior at X_new points
        """
        y_mean, K = self.get_mvn_posterior(
            self.X_train, self.y_train, X_new,
            self.nn_params, self.kernel_params, noiseless)
        y_sampled = dist.MultivariateNormal(y_mean, K).sample(rng_key, sample_shape=(n,))
        return y_mean, y_sampled

    def predict_in_batches(self, rng_key: jnp.ndarray,
                           X_new: jnp.ndarray,  batch_size: int = 100,
                           noiseless: bool = False
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new with sampled DKL hyperparameters
        by spitting the input array into chunks ("batches") and running
        self.predict on each of them one-by-one to avoid a memory overflow
        """
        predict_fn = lambda xi: self.predict(rng_key, xi, noiseless=noiseless)
        cat_dim = 1 if self.X_train.ndim == len(self.data_dim) + 2 else 0
        mean, var = self._predict_in_batches(
            rng_key, X_new, batch_size, cat_dim, predict_fn=predict_fn)
        mean = jnp.concatenate(mean, cat_dim)
        var = jnp.concatenate(var, cat_dim)
        return mean, var

    def predict(self, rng_key: jnp.ndarray, X_new: jnp.ndarray,
                params: Optional[Tuple[Dict[str, jnp.ndarray]]] = None,
                noiseless: bool = False, *args
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new points using a trained DKL model(s)

        Args:
            rng_key: random number generator key
            X_new: New ('test') data
            params: Tuple with neural network weigths and kernel parameters(optional)
            noiseless:
                Noise-free prediction. It is set to False by default as new/unseen data is assumed
                to follow the same distribution as the training data. Hence, since we introduce a model noise
                for the training data, we also want to include that noise in our prediction.

        Returns:
            Predictive mean and variance
        """

        def single_predict(x_train_i, y_train_i, x_new_i, nnpar_i, kpar_i):
            mean, cov = self.get_mvn_posterior(
                x_train_i, y_train_i, x_new_i, nnpar_i, kpar_i, noiseless)
            return mean, cov.diagonal()

        if params is None:
            nn_params = self.nn_params
            k_params = self.kernel_params
        else:
            nn_params, k_params = params

        p_args = (self.X_train, self.y_train, X_new, nn_params, k_params)
        if self.X_train.ndim == len(self.data_dim) + 2:
            mean, var = jax.vmap(single_predict)(*p_args)
        else:
            mean, var = single_predict(*p_args)

        return mean, var

    def fit_predict(self, rng_key: jnp.array, X: jnp.ndarray, y: jnp.ndarray,
                    X_new: jnp.ndarray, num_steps: int = 1000, step_size: float = 5e-3,
                    n_models: int = 1, batch_size: int = 100, noiseless: bool = False,
                    print_summary: bool = True, progress_bar=True
                    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run SVI to learn DKL model(s) parameters and make a prediction with
        trained model(s) on new data. Allows using an ensemble of models.

        Args:
            rng_key: random number generator key
            X: Input high-dimensional features
            y: Target output (scalar of vector)
            X_new: New ('test') data
            num_steps: number of SVI steps
            step_size: step size schedule for Adam optimizer
            n_models: number of models in the ensemble (defaults to 1)
            batch_size: prediction batch size (to avoid memory overflows)
            noiseless:
                Noise-free prediction. It is set to False by default as new/unseen data is assumed
                to follow the same distribution as the training data. Hence, since we introduce a model noise
                for the training data, we also want to include that noise in our prediction.
            print_summary: print summary at the end of sampling
            progress_bar: show progress bar (works only for scalar outputs)

        Returns:
            Predictive mean and variance
        """

        def single_fit_predict(key):
            self.fit(key, X, y, num_steps, step_size,
                     print_summary, progress_bar)
            mean, var = self.predict_in_batches(key, X_new, batch_size, noiseless)
            return mean, var

        keys = jax.random.split(rng_key, num=n_models)
        if n_models > 1:
            print_summary = progress_bar = 0
            mean, var = jax.vmap(single_fit_predict)(keys)
        else:
            mean, var = single_fit_predict(keys[0])

        return mean, var

    @partial(jit, static_argnames='self')
    def embed(self, X_new: jnp.ndarray) -> jnp.ndarray:
        """
        Use trained neural network(s) to embed the input data
        into the latent space(s)
        """
        def single_embed(nnpar_i, x_i):
            return self.nn_module.apply(nnpar_i, jax.random.PRNGKey(0), x_i)

        if self.X_train.ndim == len(self.data_dim) + 2:
            z = jax.vmap(single_embed)(self.nn_params, X_new)
        else:
            z = single_embed(self.nn_params, X_new)
        return z

    def _print_summary(self) -> None:
        if isinstance(self.kernel_params, dict):
            print('\nInferred GP kernel parameters')
            if self.X_train.ndim == len(self.data_dim) + 1:
                for (k, vals) in self.kernel_params.items():
                    spaces = " " * (15 - len(k))
                    print(k, spaces, jnp.around(vals, 4))
            else:
                for (k, vals) in self.kernel_params.items():
                    for i, v in enumerate(vals):
                        spaces = " " * (15 - len(k))
                        print(k+"[{}]".format(i), spaces, jnp.around(v, 4))


class MLP(hk.Module):
    """Simple MLP"""
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
