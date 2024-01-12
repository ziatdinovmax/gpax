from typing import Callable, Dict, Optional, Tuple, Type, Union

import jax
import jaxlib
import jax.numpy as jnp
import jax.random as jra
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median


from .gp import ExactGP
from .linreg import LinReg

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray]


class MeasuredNoiseGP(ExactGP):
    def __init__(self,
                 input_dim: int,
                 kernel: Union[str, kernel_fn_type],
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 lengthscale_prior_dist: Optional[dist.Distribution] = None) -> None:
        
        args = (input_dim, kernel, mean_fn, kernel_prior, mean_fn_prior, None, None, lengthscale_prior_dist)
        super(MeasuredNoiseGP, self).__init__(*args)
        self.measured_noise = None
        self.noise_predicted = None

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, measured_noise: jnp.ndarray = None, **kwargs) -> None:
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
        # Get the predictive mean and covariance
        y_mean, K = self.get_mvn_posterior(X_new, params, noiseless, **kwargs)
        # Add predicted noise to K's diagonal
        K += jnp.diag(noise_predicted)
        # Draw samples from the posterior predictive for a given set of parameters
        sig = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(rng_key, X_new.shape[:1])
        y_sampled = jnp.expand_dims(y_mean + sig, 0)
        return y_mean, y_sampled
    
    def predict(
        self,
        rng_key: jnp.ndarray,
        X_new: jnp.ndarray,
        samples: Optional[Dict[str, jnp.ndarray]] = None,
        n: int = 1,
        filter_nans: bool = False,
        noiseless: bool = False,
        device: Type[jaxlib.xla_extension.Device] = None,
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
            **jitter:
                Small positive term added to the diagonal part of a covariance
                matrix for numerical stability (Default: 1e-6)

        Returns
            Center of the mass of sampled means and all the sampled predictions
        """
        X_new = self._set_data(X_new)

        # Predict noise for X_new
        if self.noise_predicted is not None:
            noise_predicted = self.noise_predicted
        else:
            noise_predicted = self.linreg(self.X_train, self.measured_noise, X_new)
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