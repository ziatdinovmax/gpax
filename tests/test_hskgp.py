import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
from numpy.testing import assert_equal, assert_array_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.models.hskgp import VarNoiseGP
from gpax.utils import get_keys


def get_dummy_data(jax_ndarray=True, unsqueeze=False):
    X = onp.linspace(1, 2, 8) + 0.1 * onp.random.randn(8,)
    y = (10 * X**2)
    if unsqueeze:
        X = X[:, None]
    if jax_ndarray:
        return jnp.array(X), jnp.array(y)
    return X, y


@pytest.mark.parametrize("noise_kernel", ['RBF', 'Matern'])
def test_fit(noise_kernel):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = VarNoiseGP(1, 'RBF', noise_kernel=noise_kernel)
    m.fit(rng_key, X, y, num_warmup=10, num_samples=10)
    assert m.mcmc is not None


def test_get_mvn_posterior():
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1),
              "k_noise_length": jnp.array(0.5),
              "k_noise_scale": jnp.array(1.0),
              "log_var": jnp.ones(len(X))}
    m = VarNoiseGP(1, 'RBF', noise_kernel='RBF')
    m.X_train = X
    m.y_train = y
    mean, cov = m.get_mvn_posterior(X_test, params)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(cov, jnp.ndarray)
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))


def test_get_noise_samples():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = VarNoiseGP(1, 'RBF')
    m.fit(rng_key, X, y, num_warmup=10, num_samples=10)
    noise = m.get_data_var_samples()
    assert_(isinstance(noise, jnp.ndarray))
