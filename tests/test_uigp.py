import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist
from numpy.testing import assert_equal, assert_array_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.models.uigp import UIGP
from gpax.utils import get_keys


def get_dummy_data():
    X = onp.linspace(1, 2, 8) + 0.1 * onp.random.randn(8,)
    X_prime = onp.random.normal(X, 0.1)
    y = (10 * X_prime**2)
    return jnp.array(X_prime), jnp.array(y)


@pytest.mark.parametrize("n_features", [1, 5])
def test_sample_x(n_features):
    X = onp.random.randn(32, n_features)
    m = UIGP(n_features, 'RBF')
    with numpyro.handlers.seed(rng_seed=0):
        X_prime = m._sample_x(X)
    assert_(isinstance(X_prime, jnp.ndarray))
    assert_(X_prime.shape[-1], n_features)


def test_fit():
    X, y = get_dummy_data()
    m = UIGP(1, 'RBF')
    m.fit(X, y, num_warmup=10, num_samples=10)
    assert_(m.mcmc is not None)


def test_fit_with_custom_sigma_x_prior():
    X, y = get_dummy_data()
    m = UIGP(1, 'RBF', sigma_x_prior_dist=dist.HalfNormal(0.55))
    m.fit(X, y, num_warmup=10, num_samples=10)
    assert_(m.mcmc is not None)


def test_compute_gp_posterior():
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    X = X[:, None]
    X_test = X_test[:, None]
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1),
              "k_noise_length": jnp.array(0.5),
              "sigma_x": jnp.array(0.3),
              "X_prime": jnp.array(X + 0.1)
              }
    m = UIGP(1, 'RBF')
    mean, cov = m.compute_gp_posterior(X_test, X, y, params)
    assert_(isinstance(mean, jnp.ndarray))
    assert_(isinstance(cov, jnp.ndarray))
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))
