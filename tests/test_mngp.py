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
    X, y, measured_noise = get_dummy_data()
    m = MeasuredNoiseGP(1, 'RBF')
    m.fit(X, y, measured_noise, num_warmup=10, num_samples=10)
    assert m.mcmc is not None
 

@pytest.mark.parametrize("noise_pred_fn", ['linreg', 'gpreg'])
def test_fit_predict(noise_pred_fn):
    X, y, measured_noise = get_dummy_data()
    X = X[:, None]
    X_test, _, _ = get_dummy_data()
    m = MeasuredNoiseGP(1, 'RBF')
    m.fit(X, y, measured_noise, num_warmup=10, num_samples=10)
    m.measured_noise = measured_noise
    y_mean, y_var = m.predict(X_test, noise_prediction_method=noise_pred_fn)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_var, jnp.ndarray)
    assert_equal(y_mean.shape, (X_test.shape[0],))
    assert_equal(y_mean.shape, y_var.shape)
    
