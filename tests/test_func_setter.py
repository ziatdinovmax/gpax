import sys
import jax.numpy as jnp
from numpy.testing import assert_equal, assert_

sys.path.insert(0, "../gpax/")


from gpax.utils.fn import set_fn, set_kernel_fn, _set_noise_kernel_fn


def linear_kernel_test(X, Z, k_scale):
    # Dummy kernel functions for testing purposes
    return k_scale * jnp.dot(X, Z.T)


def rbf_test(X, Z, k_length, k_scale):
    # Dummy kernel functions for testing purposes
    scaled_X = X / k_length
    scaled_Z = Z / k_length
    X2 = (scaled_X ** 2).sum(1, keepdims=True)
    Z2 = (scaled_Z ** 2).sum(1, keepdims=True)
    XZ = jnp.matmul(scaled_X, scaled_Z.T)
    r2 = X2 - 2 * XZ + Z2.T

    return k_scale * jnp.exp(-0.5 * r2)


def sample_function(x, a, b):
    return a + b * x
    

def test_set_fn():
    transformed_fn = set_fn(sample_function)
    result = transformed_fn(2, {"a": 1, "b": 3})
    assert result == 7  # Expected output: 1 + 3*2 = 7


def test_set_kernel_fn():

    # Convert the dummy kernel functions
    new_linear_kernel = set_kernel_fn(linear_kernel_test)
    new_rbf = set_kernel_fn(rbf_test)

    X = jnp.array([[1, 2], [3, 4], [5, 6]])
    Z = jnp.array([[1, 2], [3, 4]])
    params_linear = {"k_scale": 1.0}
    params_rbf = {"k_length": 1.0, "k_scale": 1.0}

    # Assert the transformed function is working correctly
    assert_(jnp.array_equal(linear_kernel_test(X, Z, 1.0), new_linear_kernel(X, Z, params_linear)))
    assert_(jnp.array_equal(rbf_test(X, Z, 1.0, 1.0), new_rbf(X, Z, params_rbf)))


def test_set_kernel_fn_with_jitter():

    jitter = 1e-5

    # Convert the dummy kernel functions
    new_linear_kernel = set_kernel_fn(linear_kernel_test)
    new_rbf = set_kernel_fn(rbf_test)

    X = jnp.array([[1, 2], [3, 4], [5, 6]])
    params_linear = {"k_scale": 1.0}
    params_rbf = {"k_length": 1.0, "k_scale": 1.0}

    # Assert the transformed function is working correctly
    assert_(jnp.array_equal(linear_kernel_test(X, X, 1.0) + jitter * jnp.eye(X.shape[0]), new_linear_kernel(X, X, params_linear, jitter=jitter)))
    assert_(jnp.array_equal(rbf_test(X, X, 1.0, 1.0) + jitter * jnp.eye(X.shape[0]), new_rbf(X, X, params_rbf, jitter=jitter)))


def test_set_noise_kernel_fn():
    from gpax.kernels import RBFKernel

    X = jnp.array([[1, 2], [3, 4], [5, 6]])
    params_i = {"k_length": jnp.array([1.0]), "k_scale": jnp.array(1.0)}
    params = {"k_noise_length": jnp.array([1.0]), "k_noise_scale": jnp.array(1.0)}
    noise_rbf = _set_noise_kernel_fn(RBFKernel)
    assert_(jnp.array_equal(noise_rbf(X, X, params), RBFKernel(X, X, params_i)))
