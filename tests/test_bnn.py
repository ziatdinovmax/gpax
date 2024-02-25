import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
from numpy.testing import assert_equal, assert_array_equal

sys.path.insert(0, "../gpax/")

from gpax.models.bnn import BNN
from gpax.utils import get_keys


def get_dummy_data(feature_dim=1, target_dim=1, squeezed=False):
    X = onp.random.randn(8, feature_dim)
    y = onp.random.randn(X.shape[0], target_dim)
    if squeezed:
        return X.squeeze(), y.squeeze()
    return X, y


def test_bnn_fit():
    key, _ = get_keys()
    X, y = get_dummy_data()
    bnn = BNN(1, 1)
    bnn.fit(key, X, y, num_warmup=50, num_samples=50)
    assert bnn.mcmc is not None


def test_bnn_custom_layers_fit():
    key, _ = get_keys()
    X, y = get_dummy_data()
    bnn = BNN(1, 1, hidden_dim=[32, 16, 8])
    bnn.fit(key, X, y, num_warmup=50, num_samples=50)
    samples = bnn.get_samples()
    assert_equal(samples["w0"].shape, (50, 1, 32))
    assert_equal(samples["w1"].shape, (50, 32, 16))
    assert_equal(samples["w2"].shape, (50, 16, 8))
    assert_equal(samples["w3"].shape, (50, 8, 1))
    assert_equal(samples["b0"].shape, (50, 32))
    assert_equal(samples["b1"].shape, (50, 16))
    assert_equal(samples["b2"].shape, (50, 8))
    assert_equal(samples["b3"].shape, (50, 1))


def test_bnn_predict_with_samples():
    key, _ = get_keys()
    X_test, _ = get_dummy_data()
   
    params = {"w0": jax.random.normal(key, shape=(50, 1, 64)),
              "w1": jax.random.normal(key, shape=(50, 64, 32)),
              "w2": jax.random.normal(key, shape=(50, 32, 1)),
              "b0": jax.random.normal(key, shape=(50, 64,)),
              "b1": jax.random.normal(key, shape=(50, 32,)),
              "b2": jax.random.normal(key, shape=(50, 1,)),
              "noise": jax.random.normal(key, shape=(50,))
              }
    
    bnn = BNN(1, 1)
    f_pred, f_samples = bnn.predict(key, X_test, params)
    assert_equal(f_pred.shape, (len(X_test), 1))
    assert_equal(f_samples.shape, (50, len(X_test), 1))


def test_bnn_custom_layers_predict_custom_with_samples():
    key, _ = get_keys()
    X_test, _ = get_dummy_data()
   
    params = {"w0": jax.random.normal(key, shape=(50, 1, 32)),
              "w1": jax.random.normal(key, shape=(50, 32, 16)),
              "w2": jax.random.normal(key, shape=(50, 16, 8)),
              "w3": jax.random.normal(key, shape=(50, 8, 1)),
              "b0": jax.random.normal(key, shape=(50, 32,)),
              "b1": jax.random.normal(key, shape=(50, 16,)),
              "b2": jax.random.normal(key, shape=(50, 8,)),
              "b3": jax.random.normal(key, shape=(50, 1,)),
              "noise": jax.random.normal(key, shape=(50,))
              }
    
    bnn = BNN(1, 1, hidden_dim=[32, 16, 8])
    f_pred, f_samples = bnn.predict(key, X_test, params)
    assert_equal(f_pred.shape, (len(X_test), 1))
    assert_equal(f_samples.shape, (50, len(X_test), 1))
    

@pytest.mark.parametrize("squeezed", [True, False])
@pytest.mark.parametrize("target_dim", [1, 2])
@pytest.mark.parametrize("feature_dim", [1, 2])
def test_bnn_fit_predict(feature_dim, target_dim, squeezed):
    key, _ = get_keys()
    X, y = get_dummy_data(feature_dim, target_dim, squeezed)
    X_test, _ = get_dummy_data(feature_dim, target_dim, squeezed)
    bnn = BNN(feature_dim, target_dim, hidden_dim=[4, 2])
    bnn.fit(key, X, y, num_warmup=5, num_samples=5)
    f_pred, f_samples = bnn.predict(key, X_test)
    assert_equal(f_pred.shape, (len(X_test), target_dim))
    assert_equal(f_samples.shape, (5, len(X_test), target_dim))
