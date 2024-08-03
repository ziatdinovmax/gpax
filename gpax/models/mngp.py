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
from jax import vmap
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median


from .gp import ExactGP
from .vigp import viGP
from .linreg import LinReg
from ..utils import get_keys, put_on_device

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

        >>> # Initialize model
        >>> gp_model = gpax.MeasuredNoiseGP(input_dim=1, kernel='Matern')
        >>> # Run HMC to obtain posterior samples for the GP model parameters
        >>> gp_model.fit(X, y_mean, noise)  # X, y_mean, and noise have dimensions (n, 1), (n,), and (n,)
        >>> # Make a prediction on new inputs by extrapolating noise variance with either linear regression or gaussian process
        >>> y_pred, y_samples = gp_model.predict(X_new, noise_prediction_method='linreg')
    """
    def __init__(self,
                 input_dim: int,
                 kernel: Union[str, kernel_fn_type],
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 lengthscale_prior_dist: Optional[dist.Distribution] = None, jitter: float = 1e-6
                 ) -> None:
        args = (input_dim, kernel, mean_fn, kernel_prior, mean_fn_prior,
                None, None, lengthscale_prior_dist, jitter)
        super(MeasuredNoiseGP, self).__init__(*args)
        self.measured_noise = None
        self.noise_predicted = None

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, measured_noise: jnp.ndarray = None) -> None:
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
        k = self.kernel(X, X, kernel_params, noise, self.jitter)
        # Sample y according to the standard Gaussian process formula. Add measured noise to the covariance matrix
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k+jnp.diag(measured_noise)),
            obs=y,
        )

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        measured_noise: jnp.ndarray,
        num_warmup: int = 2000,
        num_samples: int = 2000,
        num_chains: int = 1,
        chain_method: str = "sequential",
        progress_bar: bool = True,
        print_summary: bool = True,
        device: str = None,
        rng_key: jnp.array = None
    ) -> None:
        """
        Run Hamiltonian Monter Carlo to infer the GP parameters

        Args:
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
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key: random number generator key
        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X, y = self.set_data(X, y)
        X, y, measured_noise = put_on_device(device, X, y, measured_noise)
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
        self.mcmc.run(key, X, y, measured_noise)
        if print_summary:
            self.print_summary()

    def predict(self,
                X_new: jnp.ndarray,
                noiseless: bool = True,
                device: str = None,
                noise_prediction_method: str = 'linreg',
                **kwargs
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make prediction at X_new points a trained GP model

        Args:
            X_new:
                New inputs with *(number of points, number of features)* dimensions
            noiseless:
                Noise-free prediction. It is set to False by default as new/unseen data is assumed
                to follow the same distribution as the training data. Hence, since we introduce a model noise
                by default for the training data, we also want to include that noise in our prediction.
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            noise_prediction_method:
                Method for extrapolating noise variance to new/test data.
                Choose between 'linreg' and 'gpreg'. Defaults to 'linreg'.

        Returns:
            Posterior mean and variance
        """

        if noise_prediction_method not in ["linreg", "gpreg"]:
            raise NotImplementedError(
                "For noise prediction method, select between 'linreg' and 'gpreg'")
        noise_pred_fn = self.linreg if noise_prediction_method == "linreg" else self.gpreg

        X_new = self.set_data(X_new)
        samples = self.get_samples(chain_dim=False)
        self.X_train, self.y_train, X_new, samples = put_on_device(
            device, self.X_train, self.y_train, X_new, samples)
        
        # Predict noise for X_new
        if self.noise_predicted is not None:
            noise_predicted = self.noise_predicted
        else:
            noise_predicted = noise_pred_fn(self.X_train, self.measured_noise, X_new, **kwargs)
            self.noise_predicted = noise_predicted

        predictive = lambda p: self.compute_gp_posterior(
            X_new, self.X_train, self.y_train, p, noiseless)
        # Compute predictive mean and covariance for all HMC samples
        mu_all, cov_all = vmap(predictive)(samples)
        # Calculate the average of the means
        mean_predictions = mu_all.mean(axis=0)
        # Calculate the average within-model variance and variance of the means
        average_within_model_variance = cov_all.mean(axis=0).diagonal() + jnp.clip(self.noise_predicted, 0)
        variance_of_means = jnp.var(mu_all, axis=0)
        # Total predictive variance
        total_predictive_variance = average_within_model_variance + variance_of_means

        return mean_predictions, total_predictive_variance
    
    def linreg(self, x, y, x_new, **kwargs):
        lreg = LinReg()
        lreg.train(x, y, **kwargs)
        return lreg.predict(x_new)
    
    def gpreg(self, x, y, x_new, **kwargs):
        vigp = viGP(self.kernel_dim, 'RBF', **kwargs)
        vigp.fit(x, y, progress_bar=False, print_summary=False, **kwargs)
        return vigp.predict(x_new, noiseless=True)[0]