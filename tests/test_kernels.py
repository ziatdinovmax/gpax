import sys
import pytest
import numpy as onp
import jax.numpy as jnp
from numpy.testing import assert_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.kernels import RBFKernel, MaternKernel, PeriodicKernel, index_kernel


@pytest.mark.parametrize("kernel", [RBFKernel, MaternKernel])
@pytest.mark.parametrize("dim", [1, 2])
def test_data_kernel_shapes(kernel, dim):
    x1 = onp.random.randn(5, dim)
    x2 = onp.random.randn(5, dim)
    params = {"k_length": jnp.array(1.0), "k_scale": jnp.array(1.0)}
    k = kernel(x1, x2, params)
    assert_equal(k.shape, (5, 5))


@pytest.mark.parametrize("dim", [1, 2])
def test_periodkernel_shapes(dim):
    x1 = onp.random.randn(5, dim)
    x2 = onp.random.randn(5, dim)
    params = {"k_length": jnp.array(1.0), "k_scale": jnp.array(1.0), "period": jnp.array(1.0)}
    k = PeriodicKernel(x1, x2, params)
    assert_equal(k.shape, (5, 5))


@pytest.mark.parametrize("kernel", [RBFKernel, MaternKernel])
@pytest.mark.parametrize("dim", [1, 2])
def test_data_kernel_ard_shapes(kernel, dim):
    x1 = onp.random.randn(5, dim)
    x2 = onp.random.randn(5, dim)
    params = {"k_length": jnp.ones(dim), "k_scale": jnp.array(1.0)}
    k = kernel(x1, x2, params)
    assert_equal(k.shape, (5, 5))


def test_index_kernel_shapes():
    indices1 = jnp.array([0, 1, 2])
    indices2 = jnp.array([2, 1, 0])
    params = {"W": jnp.array([[1, 2], [3, 4]]), "v": jnp.array([1, 2])}
    result = index_kernel(indices1, indices2, params)
    assert_(result.shape == (len(indices1), len(indices2)),  "Incorrect shape of result")


def test_index_kernel_shapes_uneven_obs():
    indices1 = jnp.array([1, 2])
    indices2 = jnp.array([2, 1, 0])
    params = {"W": jnp.array([[1, 2], [3, 4]]), "v": jnp.array([1, 2])}
    result = index_kernel(indices1, indices2, params)
    assert_(result.shape == (len(indices1), len(indices2)),  "Incorrect shape of result")


def test_index_kernel_computations():
    indices1 = jnp.array([0, 1])
    indices2 = jnp.array([1, 0])
    params = {"W": jnp.array([[1, 0], [0, 1]]), "v": jnp.array([1, 0])}
    result = index_kernel(indices1, indices2, params)
    expected_result = jnp.array([[0, 2], [1, 0]])
    assert_(jnp.allclose(result, expected_result), "Incorrect computations")
