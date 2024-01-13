"""
mngp.py
=======

Fully Bayesian Gaussian Process model that incorporates measured noise.

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

from typing import Callable, Dict, Optional, Tuple, Type, Union

import jax
import jaxlib
import jax.numpy as jnp
import jax.random as jra
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median


from .gp import ExactGP
from .vigp import viGP
from .linreg import LinReg
from ..utils import get_keys

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray]


class MeasuredNoiseGP(ExactGP):
    """
    Gaussian Process model that incorporates measured noise.
    This class extends the ExactGP model by allowing the inclusion of measured noise variances 
    in the GP framework. Unlike standard GP models where noise is typically inferred, this model 
    uses noise values obtained from repeated measurements at the same input points.

    Args:
        input_dim:
            Number of input dimensions
        kernel:
            Kernel function ('RBF', 'Matern', 'Periodic', or custom function)
        mean_fn:
            Optional deterministic mean function (use 'mean_fn_priors' to make it probabilistic)
        kernel_prior:
            Optional custom priors over kernel hyperparameters. Use it when passing your custom kernel.
        mean_fn_prior:
            Optional priors over mean function parameters
        lengthscale_prior_dist:
            Optional custom prior distribution over kernel lengthscale. Defaults to LogNormal(0, 1).
    
    Examples:

        >>> # Get random number generator keys for training and prediction
        >>> key1, key2 = gpax.utils.get_keys()
        >>> # Initialize model
        >>> gp_model = gpax.MeasuredNoiseGP(input_dim=1, kernel='Matern')
        >>> # Run HMC to obtain posterior samples for the GP model parameters
        >>> gp_model.fit(key1, X, y_mean, noise)  # X, y_mean, and noise have dimensions (n, 1), (n,), and (n,)
        >>> # Make a prediction on new inputs by extrapolating noise variance with either linear regression or gaussian process
        >>> y_pred, y_samples = gp_model.predict(key2, X_new, noise_prediction_method='linreg')
    """
    def __init__(self,
                 input_dim: int,
                 kernel: Union[str, kernel_fn_type],
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 lengthscale_prior_dist: Optional[dist.Distribution] = None
                 ) -> None:
        args = (input_dim, kernel, mean_fn, kernel_prior, mean_fn_prior, None, None, lengthscale_prior_dist)
        super(MeasuredNoiseGP, self).__init__(*args)
        self.measured_noise = None
        self.noise_predicted = None

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, measured_noise: jnp.ndarray = None, **kwargs) -> None:
        """GP model that accepts measured noise"""
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[0])
        # Sample kernel parameters
        if self.kernel_prior:
            kernel_params = self.kernel_prior()
        else:
            kernel_params = self._sample_kernel_params()
        # Since we provide a measured noise, we don't infer it
        noise = numpyro.deterministic("noise", jnp.array(0.0))
        # Add mean function (if any)
        if self.mean_fn is not None:
            args = [X]
            if self.mean_fn_prior is not None:
                args += [self.mean_fn_prior()]
            f_loc += self.mean_fn(*args).squeeze()
        # compute kernel (with zero noise)
        k = self.kernel(X, X, kernel_params, 0, **kwargs)
        # Sample y according to the standard Gaussian process formula. Add measured noise to the covariance matrix
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k+jnp.diag(measured_noise)),
            obs=y,
        )

    def fit(
        self,
        rng_key: jnp.array,
        X: jnp.ndarray,
        y: jnp.ndarray,
        measured_noise: jnp.ndarray,
        num_warmup: int = 2000,
        num_samples: int = 2000,
        num_chains: int = 1,
        chain_method: str = "sequential",
        progress_bar: bool = True,
        print_summary: bool = True,
        device: Type[jaxlib.xla_extension.Device] = None,
        **kwargs: float
    ) -> None:
        """
        Run Hamiltonian Monter Carlo to infer the GP parameters

        Args:
            rng_key: random number generator key
            X: 2D feature vector
            y: 1D target vector
            measured_noise: 1D vector with measured noise
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: 'sequential', 'parallel' or 'vectorized'
            progress_bar: show progress bar
            print_summary: print summary at the end of sampling
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
        self.X_train = X
        self.y_train = y
        self.measured_noise = measured_noise

        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self.model, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            jit_model_args=False,
        )
        self.mcmc.run(rng_key, X, y, measured_noise, **kwargs)
        if print_summary:
            self._print_summary()

    def _predict(
        self,
        rng_key: jnp.ndarray,
        X_new: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        noise_predicted: jnp.ndarray,
        n: int,
        noiseless: bool = False,
        **kwargs: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prediction with a single sample of GP parameters"""

        def sigma_sample(rng_key, K, X_new_shape):
            sig = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0))
            return sig * jra.normal(rng_key, X_new_shape[:1])
        
        # Get the predictive mean and covariance
        y_mean, K = self.get_mvn_posterior(X_new, params, noiseless, **kwargs)
        # Add predicted noise to K's diagonal
        K += jnp.diag(noise_predicted)
        # Draw samples from the posterior predictive for a given set of parameters
        rng_keys = jra.split(rng_key, n)
        sig = jax.vmap(sigma_sample, in_axes=(0, None, None))(rng_keys, K, X_new.shape)
        y_sampled = y_mean + sig
        return y_mean, y_sampled
    
    def predict(
        self,
        rng_key: jnp.ndarray,
        X_new: jnp.ndarray,
        samples: Optional[Dict[str, jnp.ndarray]] = None,
        n: int = 1,
        filter_nans: bool = False,
        noiseless: bool = True,
        device: Type[jaxlib.xla_extension.Device] = None,
        noise_prediction_method: str = 'linreg',
        **kwargs: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new points using posterior samples for GP parameters

        Args:
            rng_key: random number generator key
            X_new: new inputs with *(number of points, number of features)* dimensions
            samples: optional (different) samples with GP parameters
            n: number of samples from Multivariate Normal posterior for each HMC sample with GP parameters
            filter_nans: filter out samples containing NaN values (if any)
            noiseless:
                Noise-free prediction. It is set to False by default as new/unseen data is assumed
                to follow the same distribution as the training data. Hence, since we introduce a model noise
                by default for the training data, we also want to include that noise in our prediction.
            device:
                optionally specify a cpu or gpu device on which to make a prediction;
                e.g., ```device=jax.devices("gpu")[0]```
            noise_prediction_method:
                Method for extrapolating noise variance to new/test data.
                Choose between 'linreg' and 'gpreg'. Defaults to 'linreg'.
            **jitter:
                Small positive term added to the diagonal part of a covariance
                matrix for numerical stability (Default: 1e-6)

        Returns
            Center of the mass of sampled means and all the sampled predictions
        """
        if noise_prediction_method not in ["linreg", "gpreg"]:
            raise NotImplementedError(
                "For noise prediction method, select between 'linreg' and 'gpreg'")
        noise_pred_fn = self.linreg if noise_prediction_method == "linreg" else self.gpreg
        X_new = self._set_data(X_new)
        # Predict noise for X_new
        if self.noise_predicted is not None:
            noise_predicted = self.noise_predicted
        else:
            noise_predicted = noise_pred_fn(self.X_train, self.measured_noise, X_new, **kwargs)
            self.noise_predicted = noise_predicted
        if samples is None:
            samples = self.get_samples(chain_dim=False)
        if device:
            self._set_training_data(device=device)
            X_new = jax.device_put(X_new, device)
            samples = jax.device_put(samples, device)
        num_samples = len(next(iter(samples.values())))
        vmap_args = (jra.split(rng_key, num_samples), samples)
        predictive = jax.vmap(lambda prms: self._predict(prms[0], X_new, prms[1], noise_predicted, n, noiseless, **kwargs))
        y_means, y_sampled = predictive(vmap_args)
        if filter_nans:
            y_sampled_ = [y_i for y_i in y_sampled if not jnp.isnan(y_i).any()]
            y_sampled = jnp.array(y_sampled_)
        return y_means.mean(0), y_sampled
    
    def linreg(self, x, y, x_new, **kwargs):
        lreg = LinReg()
        lreg.train(x, y, **kwargs)
        return lreg.predict(x_new)
    
    def gpreg(self, x, y, x_new, **kwargs):
        keys = get_keys()
        vigp = viGP(self.kernel_dim, 'RBF', **kwargs)
        vigp.fit(keys[0], x, y, progress_bar=False, print_summary=False, **kwargs)
        return vigp.predict(keys[1], x_new, noiseless=True)[0]