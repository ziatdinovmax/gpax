import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import numpyro
from numpy.testing import assert_

sys.path.insert(0, "../gpax/")

from gpax.models.mtgp import MultiTaskGP
from gpax.utils import get_keys


def get_dummy_data():
    X = onp.linspace(1, 2, 20) + 0.1 * onp.random.randn(20,)
    y = (10 * X**2)
    return jnp.array(X), jnp.array(y)


def attach_indices(X, num_tasks):
    indices = onp.random.randint(0, num_tasks, size=len(X))
    return onp.column_stack([X, indices])


def dummy_mean_fn(x, params):
    return params["a"] * x[:, :-1]**params["b"]


def dummy_mean_fn_priors():
    a = numpyro.sample("a", numpyro.distributions.LogNormal(0, 1))
    b = numpyro.sample("b", numpyro.distributions.Normal(3, 1))
    return {"a": a, "b": b}


@pytest.mark.parametrize("num_latents", [1, 2])
@pytest.mark.parametrize("num_tasks", [2, 3])
@pytest.mark.parametrize("data_kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit_multitask(data_kernel, num_tasks, num_latents):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X = attach_indices(X, num_tasks)
    m = MultiTaskGP(1, data_kernel, num_latents=num_latents, shared_input_space=False)
    m.fit(rng_key, X, y, num_warmup=50, num_samples=50)
    assert_(isinstance(m.get_samples(), dict))


@pytest.mark.parametrize("num_latents", [1, 2])
@pytest.mark.parametrize("num_tasks", [2, 3])
@pytest.mark.parametrize("data_kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit_multivariate(data_kernel, num_tasks, num_latents):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    y = jnp.repeat(y[:, None], num_tasks, axis=1).reshape(-1)
    m = MultiTaskGP(
        1, data_kernel, num_latents=num_latents,
        num_tasks=num_tasks, shared_input_space=True)
    m.fit(rng_key, X, y, num_warmup=50, num_samples=50)
    assert_(isinstance(m.get_samples(), dict))


def test_fit_multitask_meanfn():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X = attach_indices(X, 2)
    m = MultiTaskGP(1, 'Matern', num_latents=2, shared_input_space=False,
                    mean_fn=dummy_mean_fn, mean_fn_prior=dummy_mean_fn_priors)
    m.fit(rng_key, X, y, num_warmup=50, num_samples=50)
    assert_(isinstance(m.get_samples(), dict))


def test_sample_kernel_custom_lscale_prior():
    lscale_prior_dist = numpyro.distributions.Normal(20, .1)
    m1 = MultiTaskGP(1, 'RBF', num_latents=2, num_tasks=2, rank=2)
    with numpyro.handlers.seed(rng_seed=1):
        lscale1 = m1._sample_kernel_params()["k_length"]
    m2 = MultiTaskGP(1, 'RBF', num_latents=2, num_tasks=2, rank=2, 
                     lengthscale_prior_dist=lscale_prior_dist)
    with numpyro.handlers.seed(rng_seed=1):
        lscale2 = m2._sample_kernel_params()["k_length"]
    assert_(not onp.array_equal(lscale1, lscale2))


def test_sample_task_kernel_custom_W_prior():
    W_prior_dist = numpyro.distributions.Normal(20*jnp.ones((2, 2, 2)), 0.1*jnp.ones((2, 2, 2)))
    m1 = MultiTaskGP(1, 'RBF', num_latents=2, num_tasks=2, rank=2)
    with numpyro.handlers.seed(rng_seed=1):
        W1 = m1._sample_task_kernel_params()["W"]
    m2 = MultiTaskGP(1, 'RBF', num_latents=2, num_tasks=2, rank=2,
                     W_prior_dist=W_prior_dist)
    with numpyro.handlers.seed(rng_seed=1):
        W2 = m2._sample_task_kernel_params()["W"]
    assert_(not onp.array_equal(W1, W2))


def test_sample_task_kernel_custom_v_prior():
    v_prior_dist = numpyro.distributions.Normal(20*jnp.ones((2, 2)), 0.1*jnp.ones((2, 2)))
    m1 = MultiTaskGP(1, 'RBF', num_latents=2, num_tasks=2, rank=2)
    with numpyro.handlers.seed(rng_seed=1):
        v1 = m1._sample_task_kernel_params()["v"]
    m2 = MultiTaskGP(1, 'RBF', num_latents=2, num_tasks=2, rank=2,
                     v_prior_dist=v_prior_dist)
    with numpyro.handlers.seed(rng_seed=1):
        v2 = m2._sample_task_kernel_params()["v"]
    assert_(not onp.array_equal(v1, v2))
