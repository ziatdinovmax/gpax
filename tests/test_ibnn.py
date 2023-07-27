import sys
import pytest
import numpy as onp
import jax.numpy as jnp
from numpy.testing import assert_equal


sys.path.insert(0, "../gpax/")

from gpax.models import iBNN, vi_iBNN
from gpax.utils import get_keys


def get_dummy_data(jax_ndarray=True, unsqueeze=False):
    X = onp.linspace(1, 2, 8) + 0.1 * onp.random.randn(8,)
    y = (10 * X**2)
    if unsqueeze:
        X = X[:, None]
    if jax_ndarray:
        return jnp.array(X), jnp.array(y)
    return X, y


@pytest.mark.parametrize("activation", ['erf', 'relu'])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_ibnn_fit_predict(activation, depth):
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = iBNN(1, depth=depth, activation=activation)
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    y_pred, y_sampled = m.predict(rng_keys[1], X_test)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    assert_equal(y_sampled.shape, (100, 1, X_test.shape[0]))


@pytest.mark.parametrize("activation", ['erf', 'relu'])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_viibnn_fit_predict(activation, depth):
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = vi_iBNN(1, depth=depth, activation=activation)
    m.fit(rng_keys[0], X, y, num_steps=100)
    y_pred, y_var = m.predict(rng_keys[1], X_test)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_var, jnp.ndarray)
    assert_equal(y_pred.shape, y_var.shape)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
