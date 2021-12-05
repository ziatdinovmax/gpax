from typing import Dict, Type

import jax.numpy as jnp
import jax.random as jra
import numpy as onp
import numpyro.distributions as dist

from .gp import ExactGP


def EI(rng_key: jnp.ndarray, model: Type[ExactGP],
       X: jnp.ndarray, xi: float = 0.01,
       maximize: bool = False, n: int = 1) -> jnp.ndarray:
    """
    Expected Improvement
    """
    y_mean, y_sampled = model.predict(rng_key, X, n=n)
    if n > 1:
        y_sampled = y_sampled.reshape(n * y_sampled.shape[0], -1)
    mean, sigma = y_sampled.mean(0), y_sampled.std(0)
    u = (mean - y_mean.max() - xi) / sigma
    u = -u if not maximize else u
    normal = dist.Normal(jnp.zeros_like(u), jnp.ones_like(u))
    ucdf = normal.cdf(u)
    updf = jnp.exp(normal.log_prob(u))
    return sigma * (updf + u * ucdf)


def UCB(rng_key: jnp.ndarray, model: Type[ExactGP],
        X: jnp.ndarray, beta: float = .25,
        maximize: bool = False, n: int = 1) -> jnp.ndarray:
    """
    Upper confidence bound
    """
    _, y_sampled = model.predict(rng_key, X, n=n)
    if n > 1:
        y_sampled = y_sampled.reshape(n * y_sampled.shape[0], -1)
    mean, var = y_sampled.mean(0), y_sampled.var(0)
    delta = jnp.sqrt(beta * var)
    if maximize:
        return mean + delta
    return mean - delta


def UE(rng_key: jnp.ndarray,
       model: Type[ExactGP],
       X: jnp.ndarray, n: int = 1) -> jnp.ndarray:
    """Uncertainty-based exploration (aka kriging)"""
    _, y_sampled = model.predict(rng_key, X, n=n)
    if n > 1:
        y_sampled = y_sampled.mean(1)
    return y_sampled.var(0)


def Thompson(rng_key: jnp.ndarray,
             model: Type[ExactGP],
             X: jnp.ndarray, n: int = 1) -> jnp.ndarray:
    """Thompson sampling"""
    posterior_samples = model.get_samples()
    idx = jra.randint(rng_key, (1,), 0, len(posterior_samples["k_length"]))
    samples = {k: v[idx] for (k, v) in posterior_samples.items()}
    _, tsample = model.predict(rng_key, X, samples, n)
    if n > 1:
        tsample = tsample.mean(1)
    return tsample.squeeze()


def bUCB(rng_key: jnp.ndarray, model: Type[ExactGP],
         X: jnp.ndarray, batch_size: int = 4,
         beta: float = .25,
         maximize: bool = False,
         n: int = 100,
         n_restarts: int = 20) -> jnp.ndarray:
    """
    Batch mode for the upper confidence bound
    """
    dist_all, obj_all = [], []
    for i in range(n_restarts):
        y_sampled = obtain_samples(rng_key, model, X, batch_size, n)
        mean, var = y_sampled.mean(1), y_sampled.var(1)
        delta = jnp.sqrt(beta * var)
        if maximize:
            obj = mean + delta
            points = X[obj.argmax(1)]
        else:
            obj = mean - delta
            points = X[obj.argmin(1)]
        d = get_distance(points)
        dist_all.append(d)
        obj_all.append(obj)
    idx = jnp.array(dist_all).argmax()
    return obj_all[idx]


def obtain_samples(rng_key: jnp.ndarray, model: Type[ExactGP],
                   X: jnp.ndarray, batch_size: int = 4,
                   n: int = 500) -> jnp.ndarray:
    posterior_samples = model.get_samples()
    idx = onp.arange(0, len(posterior_samples["k_length"]))
    onp.random.shuffle(idx)
    idx = idx[:batch_size]
    samples = {k: v[idx] for (k, v) in posterior_samples.items()}
    _, y_sampled = model.predict(rng_key, X, samples, n)
    return y_sampled


def get_distance(points: jnp.ndarray) -> float:
    d = []
    for p1 in points:
        for p2 in points:
            d.append(jnp.linalg.norm(p1-p2))
    return jnp.array(d).mean().item()
