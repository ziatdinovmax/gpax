"""
hypo.py
========

Utility functions for hypothesis learning based on arXiv:2112.06649

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Callable, Dict, Optional, Union

import jax.numpy as jnp
import numpy as np
import numpyro

from .models.gp import ExactGP
from .models.spm import sPM
from .utils import get_keys


def step(model: Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray],
         model_prior: Callable[[], Dict[str, jnp.ndarray]],
         X_measured: jnp.ndarray, y_measured: jnp.ndarray,
         X_unmeasured: Optional[jnp.ndarray] = None,
         gp_wrap: Optional[bool] = False,
         noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
         gp_kernel: str = 'Matern',
         gp_kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
         gp_input_dim: Optional[int] = 1,
         num_warmup: Optional[int] = 2000, num_samples: Optional[int] = 2000,
         num_chains: Optional[int] = 1,
         num_restarts: Optional[int] = 1,
         print_summary: Optional[bool] = True):
    """
    Compute model posterior and use it to derive predictive uncertainty

    Args:
        model:
            Parametric model in jax.numpy
        model_prior:
            Prior over model parameters using numpyro.distributions
        X_measured:
            Measured points
        y_measured:
            Measured values
        X_unmeasured:
            Unmeasured points
        gp_wrap:
            Wrap probabilistic model into a Gaussian process (Default: False)
        noise_prior:
            Custom prior for observation noise. Defaults to LogNormal(0,1)
        gp_kernel:
            Gaussian process kernel (if gp_wrap is True). Defaults to Matern
        gp_kernel_prior:
            Custom priors over kernel hyperparameters. Defaults to LogNormal(0,1)
        gp_input_dim:
            Number of lenghscale dimensions in GP kernel.
            Equals to number of input dimensions or 1 (default)
        num_warmup:
            Number of warmup steps for HMC. Defaults to 2000
        num_samples:
            Number of HMC samples. Defaults to 2000
        num_chains:
            Number of HMC chains. Defaults to 2000
        num_restarts:
            Number of restarts if r_hat values are not acceptable (>1.1).
            Defaults to 1
        print_summary:
            Verbose parameter

    Returns:
        Predictive uncertainty and trained model object
    """
    verbose = print_summary
    sgr = numpyro.diagnostics.split_gelman_rubin
    for i in range(num_restarts):
        rng_key, rng_key_predict = get_keys(i)
        # Get/update model posterior
        if gp_wrap:  # wrap model into a gaussian process (gives more flexibility)
            model_ = ExactGP(
                gp_input_dim, gp_kernel, model,
                gp_kernel_prior, model_prior, noise_prior)
            model_.fit(
                rng_key, X_measured, y_measured, num_warmup,
                num_samples, num_chains, print_summary=verbose)
        else:  # use a standalone model
            model_ = sPM(model, model_prior, noise_prior)
            model_.fit(
                rng_key, X_measured, y_measured, num_warmup,
                num_samples, num_chains, print_summary=verbose)
        rhats = [sgr(v).item() for (k,v) in model_.get_samples(1).items() if k != 'mu']
        if max(rhats) < 1.1:
            break
    # compute predictive uncertainty for the unmeasured part of the parameter space
    obj = 0
    if X_unmeasured is not None:
        mean, samples = model_.predict(rng_key, X_unmeasured)
        obj = samples.squeeze().var(0)
    return obj, model_


def sample_next(rewards: Union[np.array, jnp.array],
                method: Optional[str] = 'softmax',
                temperature: Optional[float] = 1.0,
                eps: Optional[float] = 0.4) -> int:
    """
    Sample model or input channel based on softmax or epsilon-greedy policy

    Args:
        rewards:
            Array of shape (N,) with running rewards
        method:
            Selection policy, choose between 'softmax' and 'epsilon-greedy'
        temperature:
            Optional temperature parameter for softmax selection policy
        eps:
            Optional epsilon parameter for epsilon-greedy policy

    Returns:
        The index of model or input channel to sample next
    """
    if method not in ['softmax', 'eps-greedy']:
        raise NotImplementedError(
            "The currently implemented samplong methods are 'softmax' and 'eps-greedy'")
    if rewards.ndim != 1:
        raise AttributeError("Pass rewards as 1-dimensional array")
    if method == 'softmax':
        idx = softmax(rewards, temperature)
    else:
        idx = eps_greedy(rewards, eps)
    return idx


def softmax(logits: Union[np.array, jnp.array],
            temperature: Optional[float] = 1.0) -> int:
    """
    Softmax selection policy. Based on
    Zai, A., Brown, B. (2020). Deep reinforcement learning in action. Manning Publications
    """
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    x = np.arange(len(logits))
    idx = np.random.choice(x, p=probs)
    return idx


def eps_greedy(rewards: Union[np.array, jnp.array],
               eps: Optional[float] = 0.4) -> int:
    """
    Epsilon-greedy selection policy. Based on
    Zai, A., Brown, B. (2020). Deep reinforcement learning in action. Manning Publications
    """
    if np.random.random() > eps:
        idx = rewards.argmax()
    else:
        idx = np.random.randint(len(rewards))
    return idx


def update_record(record: np.array, action: int, r: Union[int, float]) -> np.array:
    """
    Update the reward record. Based on
    Zai, A., Brown, B. (2020). Deep reinforcement learning in action. Manning Publications
    """
    new_r = (record[action, 0] * record[action, 1] + r) / (record[action, 0] + 1)
    record[action, 0] += 1
    record[action, 1] = new_r
    return record
