import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
from numpy.testing import assert_equal, assert_array_equal

sys.path.insert(0, "../gpax/")

from gpax.models.spm import sPM
from gpax.utils import get_keys


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


@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_fit(jax_ndarray):
    X, y = get_dummy_data(jax_ndarray)
    m = sPM(model, model_priors)
    m.fit(X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


def test_get_samples():
    X, y = get_dummy_data()
    m = sPM(model, model_priors)
    m.fit(X, y, num_warmup=100, num_samples=100)
    samples = m.get_samples()
    assert isinstance(samples, dict)
    for k, v in samples.items():
        assert isinstance(k, str)
        assert isinstance(v, jnp.ndarray)
        assert_equal(len(v), 100)


def test_prediction():
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test = onp.linspace(X.min(), X.max(), 200)
    samples = {"a": jax.random.normal(rng_keys[0], shape=(100,)),
               "b": jax.random.normal(rng_keys[0], shape=(100,)),
               "noise": jax.random.normal(rng_keys[0], shape=(100,))}
    m =sPM(model, model_priors)
    y_mean, y_sampled = m.predict(X_test, samples)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_mean.shape, X_test.squeeze().shape)
    assert_equal(y_sampled.shape, (100, X_test.shape[0]))


def test_fit_predict():
    X, y = get_dummy_data()
    X_test = onp.linspace(X.min(), X.max(), 200)
    m = sPM(model, model_priors)
    m.fit(X, y, num_warmup=100, num_samples=100)
    y_mean, y_sampled = m.predict(X_test)
    assert_equal(y_mean.shape, X_test.squeeze().shape)
    assert_equal(y_sampled.shape, (100, X_test.shape[0]))
