import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
from numpy.testing import assert_equal

sys.path.append("../../../")

from gpax.gp import ExactGP
from gpax.utils import get_keys


def get_dummy_data(jax_ndarray=True):
    X = onp.random.randn(8, 1)
    y = (10 * X**2).squeeze()
    if jax_ndarray:
        return jnp.array(X), jnp.array(y)
    return X, y


@pytest.mark.parametrize("jax_ndarray", [True, False])
@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit(kernel, jax_ndarray):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray)
    m = ExactGP(1, kernel)
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


@pytest.mark.parametrize("jax_ndarray", [True, False])
@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_get_samples(kernel, jax_ndarray):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray)
    m = ExactGP(1, kernel)
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100)
    samples = m.get_samples()
    assert isinstance(samples, dict)
    for k, v in samples.items():
        assert isinstance(k, str)
        assert isinstance(v, jnp.ndarray)
        assert_equal(len(v), 100)


@pytest.mark.parametrize("chain_dim, samples_dim", [(True, 2), (False, 1)])
def test_get_samples_chain_dim(chain_dim, samples_dim):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = ExactGP(1, 'RBF')
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100, num_chains=2)
    samples = m.get_samples(chain_dim)
    assert_equal(samples["k_scale"].ndim, samples_dim)
    assert_equal(samples["noise"].ndim, samples_dim)
    assert_equal(samples["k_length"].ndim, samples_dim + 1)


@pytest.mark.parametrize("kernel", ['RBF', 'Matern'])
def test_sample_kernel(kernel):
    m = ExactGP(1, kernel)
    with numpyro.handlers.seed(rng_seed=1):
        kernel_params = m._sample_kernel_params()
    _ = kernel_params.pop('period')
    param_names = ['k_length', 'k_scale']
    for k, v in kernel_params.items():
        assert k in param_names
        assert isinstance(v, jnp.ndarray)


def test_sample_periodic_kernel():
    m = ExactGP(1, 'Periodic')
    with numpyro.handlers.seed(rng_seed=1):
        kernel_params = m._sample_kernel_params()
    param_names = ['k_length', 'k_scale', 'period']
    for k, v in kernel_params.items():
        assert k in param_names
        assert isinstance(v, jnp.ndarray)


def test_get_mvn_posterior():
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    mean, cov = m.get_mvn_posterior(X_test, params)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(cov, jnp.ndarray)
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))


def test_single_sample_prediction():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_mean, y_sample = m._predict(rng_key, X_test, params, 1)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_sample, jnp.ndarray)
    assert_equal(y_mean.shape, X_test.squeeze().shape)
    assert_equal(y_sample.shape, X_test.squeeze().shape)


def test_prediction():
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    samples = {"k_length": jax.random.normal(rng_keys[0], shape=(100, 1)),
               "k_scale": jax.random.normal(rng_keys[0], shape=(100,)),
               "noise": jax.random.normal(rng_keys[0], shape=(100,))}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_mean, y_sampled = m.predict(rng_keys[1], X_test, samples)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_mean.shape, X_test.squeeze().shape)
    assert_equal(y_sampled.shape, (100, X_test.shape[0]))


@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit_predict(kernel):
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, kernel)
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    y_pred, y_sampled = m.predict(rng_keys[1], X_test)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    print(y_sampled.shape)
    assert_equal(y_sampled.shape, (100, X_test.shape[0]))


@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit_predict_in_batches(kernel):
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, kernel)
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    y_pred, y_sampled = m.predict_in_batches(rng_keys[1], X_test, batch_size=4)
    assert isinstance(y_pred, onp.ndarray)
    assert isinstance(y_sampled, onp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    print(y_sampled.shape)
    assert_equal(y_sampled.shape, (100, X_test.shape[0]))
