import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist
from numpy.testing import assert_equal, assert_array_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.models.hskgp import VarNoiseGP
from gpax.utils import get_keys


def get_dummy_data(unsqueeze=False):
    X = onp.linspace(1, 2, 8) + 0.1 * onp.random.randn(8,)
    y = (10 * X**2)
    if unsqueeze:
        X = X[:, None]
    return jnp.array(X), jnp.array(y)


def noise_fn(x, params):
    return params["a"] + params["b"]*x


def noise_fn_prior():
    a = numpyro.sample("a", dist.Normal(0, 1))
    b = numpyro.sample("b", dist.Normal(0, 1))
    return {"a": a, "b": b}


@pytest.mark.parametrize("noise_kernel", ['RBF', 'Matern'])
def test_fit(noise_kernel):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = VarNoiseGP(1, 'RBF', noise_kernel=noise_kernel)
    m.fit(rng_key, X, y, num_warmup=10, num_samples=10)
    assert m.mcmc is not None


def test_fit_with_custom_noise_lscale():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = VarNoiseGP(1, 'RBF', noise_lengthscale_prior_dist=dist.HalfNormal(1))
    m.fit(rng_key, X, y, num_warmup=10, num_samples=10)
    assert m.mcmc is not None


def test_fit_with_noise_mean_fn():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = VarNoiseGP(1, 'RBF', noise_mean_fn=noise_fn, noise_mean_fn_prior=noise_fn_prior)
    m.fit(rng_key, X, y, num_warmup=10, num_samples=10)
    assert m.mcmc is not None


def test_fit_with_noise_and_regular_mean_fn():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = VarNoiseGP(1, 'RBF', mean_fn = lambda x: 8*x**2,
                   noise_mean_fn=noise_fn, noise_mean_fn_prior=noise_fn_prior)
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


def test_get_mvn_posterior_with_mean_fn():
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1),
              "k_noise_length": jnp.array(0.5),
              "k_noise_scale": jnp.array(1.0),
              "log_var": jnp.ones(len(X)),
              "a": jnp.array(1.0),
              "b": jnp.array(1.0)
              }
    m = VarNoiseGP(1, 'RBF', noise_kernel='RBF', noise_mean_fn=noise_fn, noise_mean_fn_prior=noise_fn_prior)
    m.X_train = X
    m.y_train = y
    mean, cov = m.get_mvn_posterior(X_test, params)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(cov, jnp.ndarray)
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))


def test_get_mvn_posterior_with_noise_and_regular_mean_fn():
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1),
              "k_noise_length": jnp.array(0.5),
              "k_noise_scale": jnp.array(1.0),
              "log_var": jnp.ones(len(X)),
              "a": jnp.array(1.0),
              "b": jnp.array(1.0)
              }
    m = VarNoiseGP(1, 'RBF', noise_kernel='RBF',
                   mean_fn = lambda x: 8*x**2,
                   noise_mean_fn=noise_fn, noise_mean_fn_prior=noise_fn_prior)
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


def test_get_noise_samples_with_mean_fn():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = VarNoiseGP(1, 'RBF', noise_mean_fn=noise_fn, noise_mean_fn_prior=noise_fn_prior)
    m.fit(rng_key, X, y, num_warmup=10, num_samples=10)
    noise = m.get_data_var_samples()
    assert_(isinstance(noise, jnp.ndarray))
