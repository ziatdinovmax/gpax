import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import numpyro
from numpy.testing import assert_

sys.path.insert(0, "../gpax/")

from gpax.multitask.mtgp import MultiTaskGP
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
