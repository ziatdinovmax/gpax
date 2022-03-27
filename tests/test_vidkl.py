import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import haiku as hk
import numpyro
from numpy.testing import assert_equal, assert_array_equal

sys.path.insert(0, "../gpax/")

from gpax.vidkl import viDKL, MLP
from gpax.utils import get_keys


def get_dummy_data(jax_ndarray=True):
    X = onp.random.randn(21, 36)
    y = onp.random.randn(21,)
    if jax_ndarray:
        return jnp.array(X), jnp.array(y)
    return X, y


def get_dummy_image_data(jax_ndarray=True):
    X = onp.random.randn(21, 16, 16, 1)
    y = onp.random.randn(21,)
    if jax_ndarray:
        return jnp.array(X), jnp.array(y)
    return X, y


def get_dummy_vector_data(jax_ndarray=True):
    X, y = get_dummy_data(jax_ndarray)
    X = X[None].repeat(3, axis=0)
    y = y[None].repeat(3, axis=0)
    return X, y


class CustomConvNet(hk.Module):
    def __init__(self, embedim=2):
        super().__init__()
        self._embedim = embedim   

    def __call__(self, x):
        x = hk.Conv2D(32, 3)(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(2, 2, 'SAME')(x)
        x = hk.Conv2D(64, 3)(x)
        x = jax.nn.relu(x)
        x = hk.Flatten()(x)
        x = hk.Linear(self._embedim)(x)
        return x


@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_single_fit(jax_ndarray):
    X, y = get_dummy_data(jax_ndarray)
    rng_key = get_keys()[0]
    m = viDKL(X.shape[-1])
    nn_params, kernel_params, losses = m.single_fit(
        rng_key, X, y, num_steps=100, step_size=0.05)
    assert isinstance(kernel_params, dict)
    assert isinstance(nn_params, dict)
    assert isinstance(losses, jnp.ndarray)


@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_single_fit_custom_net(jax_ndarray):
    X, y = get_dummy_image_data(jax_ndarray)
    rng_key = get_keys()[0]
    m = viDKL(X.shape[1:], nn=CustomConvNet)
    nn_params, kernel_params, losses = m.single_fit(
        rng_key, X, y, num_steps=100, step_size=0.05)
    for i, val in enumerate(nn_params.values()):
        for k, v in val.items():
            if 'w' in k and i < 2:
                assert_equal(v.ndim, 4) # confirm that this is a 4-dim weights tensor of CNN


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
    mean, cov = m.get_mvn_posterior(X, y, X_test, nn_params, kernel_params)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(cov, jnp.ndarray)
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))


def test_get_mvn_posterior_noiseless():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    net = hk.transform(lambda x: MLP()(x))
    nn_params = net.init(rng_key, X)
    kernel_params = {"k_length": jnp.array([1.0]),
                     "k_scale": jnp.array(1.0),
                     "noise": jnp.array(0.1)}
    m = viDKL(X.shape[-1])
    mean1, cov1 = m.get_mvn_posterior(X, y, X_test, nn_params, kernel_params, noiseless=False)
    mean1_, cov1_ = m.get_mvn_posterior(X, y, X_test, nn_params, kernel_params, noiseless=False)
    mean2, cov2 = m.get_mvn_posterior(X, y, X_test, nn_params, kernel_params, noiseless=True)
    assert_array_equal(mean1, mean1_)
    assert_array_equal(cov1, cov1_)
    assert_array_equal(mean1, mean2)
    assert onp.count_nonzero(cov1 - cov2) > 0


def test_fit_scalar_target():
    X, y = get_dummy_data()
    rng_key = get_keys()[0]
    m = viDKL(X.shape[-1])
    m.fit(rng_key, X, y, num_steps=100, step_size=0.05)
    for v in m.kernel_params.values():
        assert v.ndim < 2
    for val in m.nn_params.values():
        for v in val.values():
            assert v.ndim < 3

def test_fit_vector_target():
    X, y = get_dummy_vector_data()
    rng_key = get_keys()[0]
    m = viDKL(X.shape[-1])
    m.fit(rng_key, X, y, num_steps=100, step_size=0.05)
    for v in m.kernel_params.values():
        assert v.ndim > 0
        assert_equal(v.shape[0], 3)
    for val in m.nn_params.values():
        for v in val.values():
            assert v.ndim > 1
            assert_equal(v.shape[0], 3)


def test_predict_scalar():
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
    mean, var = m.predict(rng_key, X_test)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(var, jnp.ndarray)
    assert_equal(mean.shape, (len(X_test),))
    assert_equal(var.shape, (len(X_test),))


def test_predict_vector():
    rng_key = get_keys()[0]
    X, y = get_dummy_vector_data()
    X_test, _ = get_dummy_vector_data()
    net = hk.transform(lambda x: MLP()(x))
    clone = lambda x: net.init(rng_key, x)
    nn_params = jax.vmap(clone)(X)
    kernel_params = {"k_length": jnp.array([[1.0], [1.0], [1.0]]),
                     "k_scale": jnp.array([1.0, 1.0, 1.0]),
                     "noise": jnp.array([0.1, 0.1, 0.1])}
    m = viDKL(X.shape[-1])
    m.X_train = X
    m.y_train = y
    m.nn_params = nn_params
    m.kernel_params = kernel_params
    mean, var = m.predict(rng_key, X_test)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(var, jnp.ndarray)
    assert_equal(mean.shape, X_test.shape[:-1])
    assert_equal(var.shape, X_test.shape[:-1])


def test_predict_in_batches_scalar():
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
    mean, var = m.predict_in_batches(rng_key, X_test, batch_size=10)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(var, jnp.ndarray)
    assert_equal(mean.shape, (len(X_test),))
    assert_equal(var.shape, (len(X_test),))


def test_predict_in_batches_vector():
    rng_key = get_keys()[0]
    X, y = get_dummy_vector_data()
    X_test, _ = get_dummy_vector_data()
    net = hk.transform(lambda x: MLP()(x))
    clone = lambda x: net.init(rng_key, x)
    nn_params = jax.vmap(clone)(X)
    kernel_params = {"k_length": jnp.array([[1.0], [1.0], [1.0]]),
                     "k_scale": jnp.array([1.0, 1.0, 1.0]),
                     "noise": jnp.array([0.1, 0.1, 0.1])}
    m = viDKL(X.shape[-1])
    m.X_train = X
    m.y_train = y
    m.nn_params = nn_params
    m.kernel_params = kernel_params
    mean, var = m.predict_in_batches(rng_key, X_test, batch_size=10)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(var, jnp.ndarray)
    assert_equal(mean.shape, X_test.shape[:-1])
    assert_equal(var.shape, X_test.shape[:-1])


def test_fit_predict_scalar():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = viDKL(X.shape[-1])
    mean, var = m.fit_predict(
        rng_key, X, y, X_test, num_steps=100, step_size=0.05, batch_size=10)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(var, jnp.ndarray)
    assert_equal(mean.shape, (len(X_test),))
    assert_equal(var.shape, (len(X_test),))


def test_fit_predict_vector():
    rng_key = get_keys()[0]
    X, y = get_dummy_vector_data()
    X_test, _ = get_dummy_vector_data()
    m = viDKL(X.shape[-1])
    mean, var = m.fit_predict(
        rng_key, X, y, X_test, num_steps=100, step_size=0.05, batch_size=10)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(var, jnp.ndarray)
    assert_equal(mean.shape, X_test.shape[:-1])
    assert_equal(var.shape, X_test.shape[:-1])


def test_fit_predict_scalar_ensemble():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = viDKL(X.shape[-1])
    mean, var = m.fit_predict(
        rng_key, X, y, X_test, n_models=4,
        num_steps=100, step_size=0.05, batch_size=10)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(var, jnp.ndarray)
    assert_equal(mean.shape, (4, len(X_test),))
    assert_equal(var.shape, (4, len(X_test),))


def test_fit_predict_vector_ensemble():
    rng_key = get_keys()[0]
    X, y = get_dummy_vector_data()
    X_test, _ = get_dummy_vector_data()
    m = viDKL(X.shape[-1])
    mean, var = m.fit_predict(
        rng_key, X, y, X_test, n_models=2,
        num_steps=100, step_size=0.05, batch_size=10)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(var, jnp.ndarray)
    assert_equal(mean.shape, (2, *X_test.shape[:-1]))
    assert_equal(var.shape, (2, *X_test.shape[:-1]))


def test_fit_predict_scalar_ensemble_custom_net():
    rng_key = get_keys()[0]
    X, y = get_dummy_image_data()
    X_test, _ = get_dummy_image_data()
    m = viDKL(X.shape[1:], nn=CustomConvNet)
    mean, var = m.fit_predict(
        rng_key, X, y, X_test, n_models=2,
        num_steps=100, step_size=0.05, batch_size=10)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(var, jnp.ndarray)
    assert_equal(mean.shape, (2, len(X_test),))
    assert_equal(var.shape, (2, len(X_test),))
