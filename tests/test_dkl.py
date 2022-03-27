import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
from numpy.testing import assert_equal, assert_array_equal

sys.path.insert(0, "../gpax/")

from gpax.dkl import DKL
from gpax.utils import get_keys


def get_dummy_data(jax_ndarray=True):
    X = onp.random.randn(10, 36)
    y = onp.random.randn(10,)
    if jax_ndarray:
        return jnp.array(X), jnp.array(y)
    return X, y


@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_fit(jax_ndarray):
    X, y = get_dummy_data(jax_ndarray)
    rng_key = get_keys()[0]
    m = DKL(X.shape[-1])
    m.fit(rng_key, X, y, num_warmup=5, num_samples=5)
    assert m.mcmc is not None


def test_get_mvn_posterior():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    params = {"w1": jax.random.normal(rng_key, shape=(36, 64)),
              "w2": jax.random.normal(rng_key, shape=(64, 32)),
              "w3": jax.random.normal(rng_key, shape=(32, 2)),
              "b1": jax.random.normal(rng_key, shape=(64,)),
              "b2": jax.random.normal(rng_key, shape=(32,)),
              "b3": jax.random.normal(rng_key, shape=(2,)),
              "k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = DKL(X.shape[-1], kernel='RBF')
    mean, cov = m._get_mvn_posterior(X, y, X_test, params)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(cov, jnp.ndarray)
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))

def test_get_mvn_posterior_noiseless():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    params = {"w1": jax.random.normal(rng_key, shape=(36, 64)),
              "w2": jax.random.normal(rng_key, shape=(64, 32)),
              "w3": jax.random.normal(rng_key, shape=(32, 2)),
              "b1": jax.random.normal(rng_key, shape=(64,)),
              "b2": jax.random.normal(rng_key, shape=(32,)),
              "b3": jax.random.normal(rng_key, shape=(2,)),
              "k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = DKL(X.shape[-1], kernel='RBF')
    mean1, cov1 = m._get_mvn_posterior(X, y, X_test, params, noiseless=False)
    mean1_, cov1_ = m._get_mvn_posterior(X, y, X_test, params, noiseless=False)
    mean2, cov2 = m._get_mvn_posterior(X, y, X_test, params, noiseless=True)
    assert_array_equal(mean1, mean1_)
    assert_array_equal(cov1, cov1_)
    assert_array_equal(mean1, mean2)
    assert onp.count_nonzero(cov1 - cov2) > 0
