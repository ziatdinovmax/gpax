import sys
import pytest
import numpy as onp
import jax.numpy as jnp
from numpy.testing import assert_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.kernels import (RBFKernel, MaternKernel, PeriodicKernel,
                          index_kernel, nngp_erf, nngp_relu, NNGPKernel,
                          MultitaskKernel, MultivariateKernel, LCMKernel)


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


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("kernel", [nngp_erf, nngp_relu])
@pytest.mark.parametrize("dim", [1, 2])
def test_nngp_shapes(kernel, dim, depth):
    x1 = onp.random.randn(1, dim)
    x2 = onp.random.randn(1, dim)
    var_b = jnp.array(1.0)
    var_w = jnp.array(1.0)
    k = kernel(x1, x2, var_b, var_w, depth)
    assert_equal(k.shape, (1,))


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("activation", ["erf", "relu"])
@pytest.mark.parametrize("dim", [1, 2])
def test_NNGPKernel(activation, dim, depth):
    x1 = onp.random.randn(5, dim)
    x2 = onp.random.randn(5, dim)
    params = {"var_b": jnp.array(1.0), "var_w": jnp.array(1.0)}
    kernel = NNGPKernel(activation, depth)
    k = kernel(x1, x2, params)
    assert_equal(k.shape, (5, 5))


def test_NNGPKernel_activations():
    x1 = onp.random.randn(5, 1)
    x2 = onp.random.randn(5, 1)
    params = {"var_b": jnp.array(1.0), "var_w": jnp.array(1.0)}
    kernel1 = NNGPKernel(activation='erf')
    k1 = kernel1(x1, x2, params)
    kernel2 = NNGPKernel(activation='relu')
    k2 = kernel2(x1, x2, params)
    assert_(not jnp.allclose(k1, k2, rtol=1e-3))


def test_MultiTaskKernel():
    base_kernel = 'RBF'
    mtkernel = MultitaskKernel(base_kernel)
    assert_(callable(mtkernel), "The result of MultitaskKernel should be a function.")


@pytest.mark.parametrize("data_kernel", [RBFKernel, MaternKernel])
@pytest.mark.parametrize("dim", [1, 2])
def test_multitask_kernel_shapes_test_noiseless(data_kernel, dim):
    x1 = onp.random.randn(5, dim)
    x2 = onp.random.randn(3, dim)
    x1 = onp.column_stack([x1, onp.zeros_like(x1)])
    x2 = onp.column_stack([x2, onp.ones_like(x2)])
    x12 = onp.vstack([x1, x2])
    params = {"k_length": jnp.array(1.0), "k_scale": jnp.array(1.0),
              "W": jnp.array([[1, 0], [0, 1]]), "v": jnp.array([1, 0])}
    mtkernel = MultitaskKernel(data_kernel)
    k = mtkernel(x12, x12, params)
    assert_equal(k.shape, (len(x12), len(x12)))


@pytest.mark.parametrize("data_kernel", [RBFKernel, MaternKernel])
@pytest.mark.parametrize("dim", [1, 2])
def test_multitask_kernel_shapes_test_noisy(data_kernel, dim):
    x1 = onp.random.randn(5, dim)
    x2 = onp.random.randn(3, dim)
    x1 = onp.column_stack([x1, onp.zeros_like(x1)])
    x2 = onp.column_stack([x2, onp.ones_like(x2)])
    x12 = onp.vstack([x1, x2])
    params = {"k_length": jnp.array(1.0), "k_scale": jnp.array(1.0),
              "W": jnp.array([[1, 0], [0, 1]]), "v": jnp.array([1, 0])}
    noise = jnp.array([1.0, 1.0])
    mtkernel = MultitaskKernel(data_kernel)
    k = mtkernel(x12, x12, params, noise)
    assert_equal(k.shape, (len(x12), len(x12)))


@pytest.mark.parametrize("data_kernel", [RBFKernel, MaternKernel])
@pytest.mark.parametrize("dim", [1, 2])
def test_multitask_kernel_shapes_train(data_kernel, dim):
    x1 = onp.random.randn(5, dim)
    x2 = onp.random.randn(5, dim)
    x1 = onp.column_stack([x1, onp.zeros_like(x1)])
    x2 = onp.column_stack([x2, onp.ones_like(x2)])
    x12 = onp.vstack([x1, x2])
    params = {"k_length": jnp.array(1.0), "k_scale": jnp.array(1.0),
              "W": jnp.array([[1, 0], [0, 1]]), "v": jnp.array([1, 0])}
    noise = jnp.array([1.0, 1.0])
    mtkernel = MultitaskKernel(data_kernel)
    k = mtkernel(x12, x12, params, noise)
    assert_equal(k.shape, (len(x12), len(x12)))


def test_MultiVariateKernel():
    base_kernel = 'RBF'
    num_tasks = 2
    mtkernel = MultivariateKernel(base_kernel, num_tasks)
    assert_(callable(mtkernel), "The result of MultiVariateKernel should be a function.")


@pytest.mark.parametrize("data_kernel", [RBFKernel, MaternKernel])
@pytest.mark.parametrize("num_tasks", [2, 3])
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("dim", [1, 2])
def test_multivariate_kernel_shapes_test_noisy(data_kernel, dim, num_tasks, rank):
    x1 = onp.random.randn(5, dim)
    x2 = onp.random.randn(3, dim)
    params = {"k_length": jnp.array(1.0), "k_scale": jnp.array(1.0),
              "W": jnp.ones((num_tasks, rank)), "v": jnp.ones(num_tasks)}
    noise = jnp.ones(num_tasks)
    mtkernel = MultivariateKernel(data_kernel, num_tasks)
    k = mtkernel(x1, x2, params, noise)
    assert_equal(k.shape, (num_tasks*len(x1), num_tasks*len(x2)))


@pytest.mark.parametrize("data_kernel", [RBFKernel, MaternKernel])
@pytest.mark.parametrize("num_tasks", [2, 3])
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("dim", [1, 2])
def test_multivariate_kernel_shapes_test_noiseless(data_kernel, dim, num_tasks, rank):
    x1 = onp.random.randn(5, dim)
    x2 = onp.random.randn(3, dim)
    params = {"k_length": jnp.array(1.0), "k_scale": jnp.array(1.0),
              "W": jnp.ones((num_tasks, rank)), "v": jnp.ones(num_tasks)}
    mtkernel = MultivariateKernel(data_kernel, num_tasks)
    k = mtkernel(x1, x2, params)
    assert_equal(k.shape, (num_tasks*len(x1), num_tasks*len(x2)))


@pytest.mark.parametrize("data_kernel", [RBFKernel, MaternKernel])
@pytest.mark.parametrize("num_tasks", [2, 3])
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("dim", [1, 2])
def test_multivariate_kernel_shapes_train(data_kernel, dim, num_tasks, rank):
    x1 = onp.random.randn(5, dim)
    x2 = onp.random.randn(5, dim)
    params = {"k_length": jnp.array(1.0), "k_scale": jnp.array(1.0),
              "W": jnp.ones((num_tasks, rank)), "v": jnp.ones(num_tasks)}
    noise = jnp.ones(num_tasks)
    mtkernel = MultivariateKernel(data_kernel, num_tasks)
    k = mtkernel(x1, x2, params, noise)
    assert_equal(k.shape, (num_tasks*len(x1), num_tasks*len(x2)))


def test_LCMKernel():
    base_kernel = 'RBF'
    lcm_kernel = LCMKernel(base_kernel)
    assert_(callable(lcm_kernel), "The result of MultitaskKernel should be a function.")


@pytest.mark.parametrize("num_latent", [1, 2])
@pytest.mark.parametrize("data_kernel", [RBFKernel, MaternKernel])
@pytest.mark.parametrize("dim", [1, 2])
def test_LCMKernel_shapes_multitask(data_kernel, dim, num_latent):
    x1 = onp.random.randn(5, dim)
    x2 = onp.random.randn(3, dim)
    x1 = onp.column_stack([x1, onp.zeros_like(x1)])
    x2 = onp.column_stack([x2, onp.ones_like(x2)])
    x12 = onp.vstack([x1, x2])
    params = {"k_length": jnp.ones(num_latent), "k_scale": jnp.ones(num_latent),
              "W": jnp.ones((num_latent, 2, 2)), "v": jnp.ones((num_latent, 2))}
    noise = jnp.array([1.0, 1.0])
    mtkernel = LCMKernel(data_kernel, shared_input_space=False)
    k = mtkernel(x12, x12, params, noise)
    assert_equal(k.shape, (len(x12), len(x12)))


@pytest.mark.parametrize("num_latent", [1, 2])
@pytest.mark.parametrize("data_kernel", [RBFKernel, MaternKernel])
@pytest.mark.parametrize("num_tasks", [2, 3])
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("dim", [1, 2])
def test_LCMKernel_shapes_multivariate(data_kernel, dim, num_latent, rank, num_tasks):
    x1 = onp.random.randn(5, dim)
    x2 = onp.random.randn(3, dim)
    params = {"k_length": jnp.ones(num_latent), "k_scale": jnp.ones(num_latent),
              "W": jnp.ones((num_latent, num_tasks, rank)), "v": jnp.ones((num_latent, num_tasks))}
    noise = jnp.ones(num_tasks)
    mtkernel = LCMKernel(data_kernel, shared_input_space=True, num_tasks=num_tasks)
    k = mtkernel(x1, x2, params, noise)
    assert_equal(k.shape, (num_tasks*len(x1), num_tasks*len(x2)))
