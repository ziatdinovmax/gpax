import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import numpyro
from numpy.testing import assert_

sys.path.insert(0, "../gpax/")

from gpax.hypo import step, sample_next


def get_dummy_data(jax_ndarray=True):
    X = onp.linspace(1, 2, 8) + 0.1 * onp.random.randn(8,)
    y = (10 * X**2)
    if jax_ndarray:
        return jnp.array(X), jnp.array(y)
    return X, y


def model(x, params):
    return params["a"] * x**params["b"]


def model_priors():
    a = numpyro.sample("a", numpyro.distributions.LogNormal(0, 1))
    b = numpyro.sample("b", numpyro.distributions.Normal(3, 1))
    return {"a": a, "b": b}


@pytest.mark.parametrize("method", ['softmax', 'eps-greedy'])
def test_sample_next(method):
    rewards = onp.array([0.0, 0.1, 0.2])
    idx = sample_next(rewards, method)
    assert_(isinstance(idx, (onp.int64, int)))


@pytest.mark.parametrize("gp_wrap", [True, False])
def test_step(gp_wrap):
    X, y = get_dummy_data()
    obj, _ = step(
        model, model_priors, X, y, X, gp_wrap=gp_wrap,
        num_warmup=50, num_samples=50)
    assert_(isinstance(obj, jnp.ndarray))
