import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
from numpy.testing import assert_equal, assert_array_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.models.vigp import viGP
from gpax.utils import get_keys, enable_x64

enable_x64()


def get_dummy_data(jax_ndarray=True, unsqueeze=False):
    X = onp.linspace(1, 2, 8) + 0.1 * onp.random.randn(8,)
    y = (10 * X**2)
    if unsqueeze:
        X = X[:, None]
    if jax_ndarray:
        return jnp.array(X), jnp.array(y)
    return X, y


def dummy_mean_fn(x, params):
    return params["a"] * x**params["b"]


def dummy_mean_fn_priors():
    a = numpyro.sample("a", numpyro.distributions.LogNormal(0, 1))
    b = numpyro.sample("b", numpyro.distributions.Normal(3, 1))
    return {"a": a, "b": b}


def gp_kernel_custom_prior():
    length = numpyro.sample("k_length", numpyro.distributions.Uniform(0, 1))
    scale = numpyro.sample("k_scale", numpyro.distributions.LogNormal(0, 1))
    return {"k_length": length, "k_scale": scale}


@pytest.mark.parametrize("jax_ndarray", [True, False])
@pytest.mark.parametrize("unsqueeze", [True, False])
@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit(kernel, jax_ndarray, unsqueeze):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray, unsqueeze)
    m = viGP(1, kernel)
    m.fit(rng_key, X, y, num_steps=100)
    assert m.svi is not None


@pytest.mark.parametrize("jax_ndarray", [True, False])
@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_get_samples(kernel, jax_ndarray):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray)
    m = viGP(1, kernel)
    m.fit(rng_key, X, y, num_steps=100)
    samples = m.get_samples()
    assert isinstance(samples, dict)
    for k, v in samples.items():
        assert isinstance(k, str)
        assert isinstance(v, jnp.ndarray)


@pytest.mark.parametrize("unsqueeze", [True, False])
def test_prediction(unsqueeze):
    rng_keys = get_keys()
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=unsqueeze)
    params = {"k_length": jax.random.normal(rng_keys[0], shape=(1, 1)),
              "k_scale": jax.random.normal(rng_keys[0], shape=(1,)),
               "noise": jax.random.normal(rng_keys[0], shape=(1,))}
    m = viGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_mean, y_var = m.predict(rng_keys[1], X_test, params)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_var, jnp.ndarray)
    assert_equal(y_mean.shape, X_test.squeeze().shape)
    assert_equal(y_var.shape, X_test.squeeze().shape)


def test_noiseless_prediction():
    rng_keys = get_keys()
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    samples = {"k_length": jax.random.normal(rng_keys[0], shape=(1, 1)),
               "k_scale": jax.random.normal(rng_keys[0], shape=(1,)),
               "noise": jax.random.normal(rng_keys[0], shape=(1,))}
    m = viGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_mean1, y_var1 = m.predict(rng_keys[1], X_test, samples, noiseless=True)
    y_mean2, y_var2 = m.predict(rng_keys[1], X_test, samples, noiseless=False)
    assert_array_equal(y_mean1, y_mean2)
    assert onp.count_nonzero(y_var1 - y_var2) > 0


@pytest.mark.parametrize("unsqueeze", [True, False])
@pytest.mark.parametrize("batch_size", [2, 3, 8])
def test_prediction_in_batches(unsqueeze, batch_size):
    rng_keys = get_keys()
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=unsqueeze)
    params = {"k_length": jax.random.normal(rng_keys[0], shape=(1, 1)),
              "k_scale": jax.random.normal(rng_keys[0], shape=(1,)),
              "noise": jax.random.normal(rng_keys[0], shape=(1,))}
    m = viGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_mean, y_var = m.predict_in_batches(
        rng_keys[1], X_test, batch_size, params)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_var, jnp.ndarray)
    assert_equal(y_mean.shape, X_test.squeeze().shape)
    assert_equal(y_var.shape, X_test.squeeze().shape)


@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit_predict(kernel):
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = viGP(1, kernel)
    m.fit(rng_keys[0], X, y, num_steps=100)
    y_pred, y_var = m.predict(rng_keys[1], X_test)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_var, jnp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    assert_equal(y_var.shape, X_test.squeeze().shape)


@pytest.mark.parametrize("batch_size", [2, 3, 8])
def test_fit_predict_in_batches(batch_size):
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = viGP(1, "RBF")
    m.fit(rng_keys[0], X, y, num_steps=100)
    y_mean, y_var = m.predict_in_batches(
        rng_keys[1], X_test, batch_size)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_var, jnp.ndarray)
    assert_equal(y_mean.shape, X_test.squeeze().shape)
    assert_equal(y_var.shape, X_test.squeeze().shape)



@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_fit_with_mean_fn(jax_ndarray):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray)
    m = viGP(1, 'RBF', mean_fn = lambda x: 8*x**2)
    m.fit(rng_key, X, y, num_steps=100)
    assert m.svi is not None


@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_fit_with_prob_mean_fn(jax_ndarray):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray)
    m = viGP(1, 'RBF', mean_fn=dummy_mean_fn, mean_fn_prior=dummy_mean_fn_priors)
    m.fit(rng_key, X, y, num_steps=100)
    assert m.svi is not None


def test_fit_predict_with_mean_fn():
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = viGP(1, 'RBF', mean_fn = lambda x: 8*x**2)
    m.fit(rng_keys[0], X, y, num_steps=100)
    y_pred, y_var = m.predict(rng_keys[1], X_test)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_var, jnp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    assert_equal(y_var.shape, X_test.squeeze().shape)


def test_fit_predict_with_prob_mean_fn():
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = viGP(1, 'RBF', mean_fn=dummy_mean_fn, mean_fn_prior=dummy_mean_fn_priors)
    m.fit(rng_keys[0], X, y, num_steps=100)
    y_pred, y_var = m.predict(rng_keys[1], X_test)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_var, jnp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    assert_equal(y_var.shape, X_test.squeeze().shape)


def test_sample_from_prior():
    rng_key, _ = get_keys()
    X, _ = get_dummy_data()
    m = viGP(1, 'RBF')
    prior_pred = m.sample_from_prior(rng_key, X, num_samples=8)
    assert_equal(prior_pred.shape, (8, X.shape[0]))


def test_jitter_fit():
    rng_key, _ = get_keys()
    X, y = get_dummy_data()
    m1 = viGP(1, 'RBF')
    m1.fit(rng_key, X, y, num_steps=100, jitter=1e-6)
    samples1 = m1.get_samples()
    m2 = viGP(1, 'RBF')
    m2.fit(rng_key, X, y, num_steps=100, jitter=1e-6)
    samples1a = m2.get_samples()
    m3 = viGP(1, 'RBF')
    m3.fit(rng_key, X, y, num_steps=100, jitter=1e-4)
    samples2 = m3.get_samples()
    assert_(samples1["k_length"] - samples1a["k_length"] == 0)
    assert_(samples1["k_length"] - samples2["k_length"] != 0)


def test_jitter_predict():
    rng_keys = get_keys()
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    samples = {"k_length": jax.random.normal(rng_keys[0], shape=(1, 1)),
               "k_scale": jax.random.normal(rng_keys[0], shape=(1,)),
               "noise": jax.random.normal(rng_keys[0], shape=(1,))}
    m = viGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_mean1, y_var1 = m.predict(rng_keys[1], X_test, samples, jitter=1e-6)
    y_mean2, y_var2 = m.predict(rng_keys[1], X_test, samples, jitter=1e-4)
    assert_(onp.count_nonzero(y_var1 - y_var2) > 0)
    assert_(onp.count_nonzero(y_mean1 - y_mean2) > 0)


def test_guide_type():
    rng_key, _ = get_keys()
    X, y = get_dummy_data()
    m = viGP(1, 'RBF',  guide='delta')
    m.fit(rng_key, X, y, num_steps=100,)
    samples1 = m.get_samples()
    m = viGP(1, 'RBF',  guide='delta')
    m.fit(rng_key, X, y, num_steps=100)
    samples1a = m.get_samples()
    m = viGP(1, 'RBF',  guide='normal')
    m.fit(rng_key, X, y, num_steps=100)
    samples2 = m.get_samples()
    assert_(samples1["k_length"] - samples1a["k_length"] == 0)
    assert_(samples1["k_length"] - samples2["k_length"] != 0)
