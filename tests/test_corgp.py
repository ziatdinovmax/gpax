import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import numpyro
from numpy.testing import assert_

sys.path.insert(0, "../gpax/")

from gpax.models.corgp import CoregGP
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


@pytest.mark.parametrize("num_tasks", [2, 3])
@pytest.mark.parametrize("data_kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit_corgp(data_kernel, num_tasks):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X = attach_indices(X, num_tasks)
    m = CoregGP(1, data_kernel)
    m.fit(rng_key, X, y, num_warmup=50, num_samples=50)
    assert_(isinstance(m.get_samples(), dict))


def test_fit_corgp_meanfn():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X = attach_indices(X, 2)
    m = CoregGP(1, 'Matern', mean_fn=dummy_mean_fn, mean_fn_prior=dummy_mean_fn_priors)
    m.fit(rng_key, X, y, num_warmup=50, num_samples=50)
    assert_(isinstance(m.get_samples(), dict))
