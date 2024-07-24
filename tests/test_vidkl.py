import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import haiku as hk
from numpy.testing import assert_equal, assert_array_equal

sys.path.insert(0, "../gpax/")

from gpax.models.vidkl import viDKL
from gpax.utils import get_keys
from gpax.models.nets import HaikuMLP


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
    y = y[None].repeat(3, axis=0)
    return X, y


def get_transformed_model(z_dim=2):
    return hk.transform(lambda x: HaikuMLP([32, 16, 8], z_dim, 'tanh')(x))


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
def test_fit(jax_ndarray):
    X, y = get_dummy_data(jax_ndarray)
    m = viDKL(X.shape[-1])
    m.fit(
        X, y, num_steps=100, step_size=0.05)
    assert m.params is not None


def test_fit_custom_net():
    X, y = get_dummy_image_data()
    m = viDKL(X.shape[1:], nn=CustomConvNet)
    m.fit(X, y, num_steps=100, step_size=0.05)
    assert m.params is not None
    

def test_compute_gp_posterior_posterior():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    transformed_model = get_transformed_model()
    params = transformed_model.init(rng_key, X)
    params["k_length"] = jnp.array([1.0])
    params["k_scale"] = jnp.array(1.0)
    params["noise"] = jnp.array(0.1)

    m = viDKL(X.shape[-1], kernel='RBF')
    mean, cov = m.compute_gp_posterior(X_test, X, y, params)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(cov, jnp.ndarray)
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))


@pytest.mark.parametrize("z_dim", [2, 3])
def test_fit_embed(z_dim):
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = viDKL(X.shape[-1], z_dim)
    m.fit(X, y, num_steps=100, step_size=0.05)
    z = m.embed(X_test)
    assert_equal(z.shape[1], X.shape[0])
    assert_equal(z.shape[2], z_dim)
