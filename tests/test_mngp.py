import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist
from numpy.testing import assert_equal, assert_array_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.models.mngp import MeasuredNoiseGP
from gpax.utils import get_keys


def variable_noise(x):
    return 0.1 + 0.5 * x

def get_dummy_data():
    f = lambda x: onp.sin(x) * x
    X = onp.linspace(1, 2, 8)
    y_all = onp.array([f(x) + onp.random.normal(0, variable_noise(x), 10) for x in X])
    y = y_all.mean(1)
    measured_noise = y_all.var(1)
    return jnp.array(X), jnp.array(y), jnp.array(measured_noise)


def test_fit():
    rng_key = get_keys()[0]
    X, y, measured_noise = get_dummy_data()
    m = MeasuredNoiseGP(1, 'RBF')
    m.fit(rng_key, X, y, measured_noise, num_warmup=10, num_samples=10)
    assert m.mcmc is not None


def test_get_mvn_posterior():
    X, y, measured_noise = get_dummy_data()
    X_test, _, _ = get_dummy_data()
    X = X[:, None]
    X_test = X_test[:, None]
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.0),
              }
    m = MeasuredNoiseGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    m.measured_noise = measured_noise
    mean, cov = m.get_mvn_posterior(X_test, params)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(cov, jnp.ndarray)
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))


@pytest.mark.parametrize("n", [1, 5])
def test_predict_single_sample(n):
    key = get_keys()[0]
    X, y, measured_noise = get_dummy_data()
    X_test, _, _ = get_dummy_data()
    X = X[:, None]
    X_test = X_test[:, None]
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.0),
              }
    m = MeasuredNoiseGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    m.measured_noise = measured_noise
    noise_predicted = 0.5 * X_test
    mean, sample = m._predict(key, X_test, params, noise_predicted, n)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(sample, jnp.ndarray)
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(sample.shape, (n, X_test.shape[0]))
 

@pytest.mark.parametrize("noise_pred_fn", ['linreg', 'gpreg'])
def test_predict(noise_pred_fn):
    rng_keys = get_keys()
    X, y, measured_noise = get_dummy_data()
    X = X[:, None]
    X_test, _, _ = get_dummy_data()
    samples = {"k_length": jax.random.normal(rng_keys[0], shape=(100, 1)),
               "k_scale": jax.random.normal(rng_keys[0], shape=(100,)),
               "noise": jax.random.normal(rng_keys[0], shape=(100,))}
    m = MeasuredNoiseGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    m.measured_noise = measured_noise
    y_mean, y_sampled = m.predict(rng_keys[1], X_test, samples, noise_prediction_method=noise_pred_fn)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_mean.shape, (X_test.shape[0],))
    assert_equal(y_sampled.shape, (100, 1, X_test.shape[0]))
    
