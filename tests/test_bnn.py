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


def get_dummy_data(length=8, feature_dim=1, target_dim=1, squeezed=False):
    X = onp.random.randn(length, feature_dim)
    y = onp.random.randn(X.shape[0], target_dim)
    if squeezed:
        return X.squeeze(), y.squeeze()
    return X, y


def test_bnn_fit():
    X, y = get_dummy_data()
    bnn = BNN(1, 1)
    bnn.fit(X, y, num_warmup=50, num_samples=50)
    assert bnn.mcmc is not None


def test_bnn_custom_layers_fit():
    X, y = get_dummy_data()
    bnn = BNN(1, 1, hidden_dim=[32, 16])
    bnn.fit(X, y, num_warmup=5, num_samples=5)
    samples = bnn.get_samples()
    assert_equal(samples["feature_extractor/haiku_mlp/Dense0.w"].shape, (5, 1, 32))
    assert_equal(samples["feature_extractor/haiku_mlp/Dense1.w"].shape, (5, 32, 16))
    assert_equal(samples["feature_extractor/haiku_mlp/Dense2.w"].shape, (5, 16, 1))
    assert_equal(samples["feature_extractor/haiku_mlp/Dense0.b"].shape, (5, 32))
    assert_equal(samples["feature_extractor/haiku_mlp/Dense1.b"].shape, (5, 16))
    assert_equal(samples["feature_extractor/haiku_mlp/Dense2.b"].shape, (5, 1))


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
    f_pred, f_var = bnn.predict(X_test, params)
    assert_equal(f_pred.shape, (len(X_test), 1))
    assert_equal(f_pred.shape, f_var.shape)


@pytest.mark.parametrize("squeezed", [True, False])
@pytest.mark.parametrize("target_dim", [1, 2])
@pytest.mark.parametrize("feature_dim", [1, 2])
def test_bnn_fit_predict(feature_dim, target_dim, squeezed):
    X, y = get_dummy_data(8, feature_dim, target_dim, squeezed)
    X_test, _ = get_dummy_data(8, feature_dim, target_dim, squeezed)
    bnn = BNN(feature_dim, target_dim, hidden_dim=[4, 2])
    bnn.fit(X, y, num_warmup=5, num_samples=5)
    f_pred, f_var = bnn.predict(X_test)
    assert_equal(f_pred.shape, (len(X_test), target_dim))
    assert_equal(f_pred.shape, f_var.shape)


def test_bnn_predict_in_batches():
    X, y = get_dummy_data(8, 5, 1)
    X_test, _ = get_dummy_data(50, 5, 1)
    bnn = BNN(5, 1, hidden_dim=[4, 2])
    bnn.fit(X, y, num_warmup=5, num_samples=5)
    f_pred, f_var = bnn.predict(X_test)
    f_pred_b, f_var_b = bnn.predict_in_batches(X_test, batch_size=20)

    assert_array_equal(f_pred, f_pred_b)
    assert_array_equal(f_var, f_var_b)
