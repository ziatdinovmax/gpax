import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
from numpy.testing import assert_equal

sys.path.insert(0, "../gpax/")

from gpax.models.sparse_gp import viSparseGP
from gpax.utils import get_keys, enable_x64

enable_x64()


def get_dummy_data(jax_ndarray=True, unsqueeze=False):
    X = onp.linspace(1, 2, 50) + 0.1 * onp.random.randn(50,)
    y = (10 * X**2)
    if unsqueeze:
        X = X[:, None]
    if jax_ndarray:
        return jnp.array(X), jnp.array(y)
    return X, y


@pytest.mark.parametrize("jax_ndarray", [True, False])
@pytest.mark.parametrize("unsqueeze", [True, False])
def test_fit(jax_ndarray, unsqueeze):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray, unsqueeze)
    m = viSparseGP(1, 'Matern')
    m.fit(rng_key, X, y, num_steps=100)
    assert m.svi is not None
    assert isinstance(m.Xu, jnp.ndarray)


def test_inducing_points_optimization():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m1 = viSparseGP(1, 'Matern')
    m1.fit(rng_key, X, y, num_steps=1)
    m2 = viSparseGP(1, 'Matern')
    m2.fit(rng_key, X, y, num_steps=100)
    assert not jnp.array_equal(m1.Xu, m2.Xu)
    
   
def test_get_mvn_posterior():
    rng_keys = get_keys()
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    params = {"k_length": jax.random.normal(rng_keys[0], shape=(1, 1)),
              "k_scale": jax.random.normal(rng_keys[0], shape=(1,)),
               "noise": jax.random.normal(rng_keys[0], shape=(1,))}
    m = viSparseGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    m.Xu = X[::2].copy()

    mean, cov = m.get_mvn_posterior(X_test, params)

    assert isinstance(mean, jnp.ndarray)
    assert isinstance(cov, jnp.ndarray)
    assert_equal(mean.shape, X_test.squeeze().shape)
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))