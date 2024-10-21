from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import haiku as hk
import jax.numpy as jnp
import numpy as onp
import numpyro
import numpyro.distributions as dist
from jax import lax, random, vmap, jit
from numpyro.contrib.module import random_haiku_module
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal

from ..kernels import LCMKernel
from ..utils.utils import get_haiku_dict
from .gp import ExactGP
from .vidkl import MLP


class viMPDKL:
    def __init__(self, input_dim: List[Tuple[int]], z_dim: int = 2,
                 nets: List[Callable[[jnp.ndarray], jnp.ndarray]] = None,
                 data_kernel: str = 'RBF',
                 num_latents: int = None,
                 rank: Optional[int] = None,
                 data_kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 guide: str = 'delta', task_kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior_dist: Optional[dist.Distribution] = None,
                 lengthscale_prior_dist: Optional[dist.Distribution] = None,
                 W_prior_dist: Optional[dist.Distribution] = None,
                 v_prior_dist: Optional[dist.Distribution] = None,
                 **kwargs) -> None:
        self.input_dim = input_dim
        self.kernel_dim = z_dim
        self.nn_modules = [hk.transform(lambda x: nn(z_dim)(x)) for nn in nets]  # list with a separate neural net for each task
        self.num_tasks = len(nets)
        self.num_latents = num_latents
        if self.num_latents is None:
            self.num_latents = self.num_tasks
        self.rank = rank
        if self.rank is None:
            self.rank = self.num_tasks - 1
        self.kernel = LCMKernel(
            data_kernel, False, **kwargs)

        self.data_kernel_prior = data_kernel_prior
        self.task_kernel_prior = task_kernel_prior
        self.noise_prior_dist = noise_prior_dist
        self.lengthscale_prior_dist = lengthscale_prior_dist
        self.W_prior_dist = W_prior_dist
        self.v_prior_dist = v_prior_dist
        self.guide_type = AutoNormal if guide == 'normal' else AutoDelta
        
        self.kernel_params = None
        self.nn_params = None
        self.X_train = None
        self.y_train = None
        self.mcmc = None

    def model(self, X: List[jnp.ndarray], y: jnp.ndarray = None, **kwargs) -> None:
        """DKL probabilistic model"""

        # Initialize GP kernel mean function at zeros
        f_loc = jnp.zeros(
            sum([x.shape[0] for x in X])
        )

        # Initialize NNs
        nets = []  # each "task" has its own neural net
        for n in range(self.num_tasks):
            nets.append(random_haiku_module(
                "feature_extractor_{}".format(n), self.nn_modules[n], input_shape=(1, *self.input_dim[n]),
                prior=(lambda name, shape: dist.Cauchy() if name.startswith("b") else dist.Normal())))
            
        # Apply NNs to data  (can be included in a previous loop)
        z = []
        for i, net in enumerate(nets):
            z_i = net(X[i])
            task_id = jnp.ones(len(z_i)) * i  # assign indices
            z_i = jnp.column_stack([z_i, task_id])  # add indices as the last column
            z.append(z_i)
        # Concatenate all the latent embeddings
        z = jnp.concatenate(z, axis=0)


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

        # Sample noise  - this should be a separate method
        noise = self._sample_noise()

        # Compute multitask_kernel
        k = self.kernel(z, z, kernel_params, noise, **kwargs)

        # Sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def fit(self, rng_key: jnp.array, X: List[jnp.ndarray], y: jnp.ndarray,
            num_steps: int = 1000, step_size: float = 5e-3,
            print_summary: bool = True, progress_bar=True,
            **kwargs) -> None:
        """
        Run stochastic variational inference to learn a DKL model parameters
        """
        self.X_train = X
        self.y_train = y

        # Setup optimizer and SVI
        optim = numpyro.optim.Adam(step_size=step_size, b1=0.5)
        svi = SVI(
            self.model,
            guide=self.guide_type(self.model),
            optim=optim,
            loss=Trace_ELBO(),
            X=X,
            y=y,
            **kwargs
        )
        params, _, losses = svi.run(rng_key, num_steps, progress_bar=progress_bar)
        # Get DKL parameters from the guide
        params_map = svi.guide.median(params)
        print(params_map)
        # Get NN weights
        nn_params = []  # each element i in the list is a dictionary with the params of the NN used for task i
        for i in range(self.num_tasks):  # consider having this as utility func (by modifying the get_haiku_dict)
            nn_params.append(get_haiku_dict(
                {k: v for (k, v) in params_map.items()
                 if k.startswith('feature_extractor_{}'.format(i))}))
        self.nn_params = nn_params
        # Get GP kernel hyperparmeters
        self.kernel_params = {k: v for (k, v) in params_map.items() 
                              if not k.startswith("feature_extractor")}
        # if print_summary:
        #     self._print_summary()
        
    #@partial(jit, static_argnames='self')
    def get_mvn_posterior(self,
                          X_train: jnp.ndarray,
                          y_train: jnp.ndarray,
                          X_new: jnp.ndarray,
                          nn_params: Dict[str, jnp.ndarray],
                          k_params: Dict[str, jnp.ndarray],
                          noiseless: bool = False,
                          **kwargs
                          ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns predictive mean and covariance at new points
        (mean and cov, where cov.diagonal() is 'uncertainty')
        given a single set of DKL parameters
        """
        noise = k_params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        # embed data into the latent space with a separate NN for each task
        task_functions = [self.create_task_function(params) for params in nn_params]
        apply_nn = lambda x, task_index: lax.switch(task_index, task_functions, x)
        z_train = vmap(apply_nn)(X_train[:, :-1], X_train[:, -1])
        z_test = vmap(apply_nn)(X_new[:, :-1], X_new[:, -1])
        # append the task indices tot he embeddings
        z_train = jnp.column_stack((z_train, X_train[:, -1]))
        z_test = jnp.column_stack((z_test, X_new[:, -1]))
        # compute kernel matrices for train and test data
        k_pp = self.kernel(z_test, z_test, k_params, noise_p, **kwargs)
        k_pX = self.kernel(z_test, z_train, k_params, jitter=0.0)
        k_XX = self.kernel(z_train, z_train, k_params, noise, **kwargs)
        # compute the predictive covariance and mean
        K_xx_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_train))
        return mean, cov

    def create_task_function(self, params):
        """Create an embedding function for each task"""
        return lambda x: self.nn_module.apply(params, random.PRNGKey(0), x)

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
