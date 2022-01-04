import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import haiku as hk
import numpyro
from numpy.testing import assert_equal

sys.path.append("../../../")

from gpax.vidkl import viDKL, MLP
from gpax.utils import get_keys


def get_dummy_data(jax_ndarray=True):
    X = onp.random.randn(10, 36)
    y = onp.random.randn(10,)
    if jax_ndarray:
        return jnp.array(X), jnp.array(y)
    return X, y


def get_dummy_nn_params():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    mlp = MLP()
    net = hk.transform(lambda x: mlp()(x))
    params = net.init(rng_key, X, y)


@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_fit(jax_ndarray):
    X, y = get_dummy_data(jax_ndarray)
    rng_key = get_keys()[0]
    m = viDKL(X.shape[-1])
    m.fit(rng_key, X, y, num_steps=100, step_size=0.05)
    assert m.kernel_params is not None
    assert m.nn_params is not None


def test_get_mvn_posterior():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    net = hk.transform(lambda x: MLP()(x))
    nn_params = net.init(rng_key, X)
    kernel_params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = viDKL(X.shape[-1])
    m.X_train = X
    m.y_train = y
    m.nn_params = nn_params
    m.kernel_params = kernel_params
    mean, cov = m.get_mvn_posterior(X_test)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(cov, jnp.ndarray)
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))


def test_predict():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    net = hk.transform(lambda x: MLP()(x))
    nn_params = net.init(rng_key, X)
    kernel_params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = viDKL(X.shape[-1])
    m.X_train = X
    m.y_train = y
    m.nn_params = nn_params
    m.kernel_params = kernel_params
    y_mean, y_sampled = m.predict(rng_key, X_test, n=100)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_mean.shape, (len(X_test),))
    assert_equal(y_sampled.shape, (100, len(X_test)))
