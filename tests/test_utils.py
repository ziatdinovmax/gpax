import sys
import pytest
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jra
import numpyro
from numpy.testing import assert_equal, assert_, assert_array_equal

sys.path.insert(0, "../gpax/")

from gpax.utils import preprocess_sparse_image, split_dict, random_sample_dict, get_keys, initialize_inducing_points


def test_sparse_img_processing():
    img = onp.random.randn(16, 16)
    # Generate random indices
    idx = [onp.random.randint(0, 16) for _ in range(100)], [onp.random.randint(0, 16) for _ in range(100)]
    # Set these indices to zero
    img[idx] = 0
    # Test the utility function
    X, y, X_full = preprocess_sparse_image(img)
    assert_equal(X.ndim, 2)
    assert_equal(y.ndim,  1)
    assert_equal(X_full.ndim, 2)
    assert_(X.shape[0] < 16*16)
    assert_equal(X.shape[1], 2)
    assert_equal(y.shape[0], X.shape[0])
    assert_equal(X_full.shape[0], 16*16)
    assert_equal(X_full.shape[1], 2)


def test_split_dict():
    data = {
        'a': jnp.array([1, 2, 3, 4, 5, 6]),
        'b': jnp.array([10, 20, 30, 40, 50, 60])
    }
    chunk_size = 4

    result = split_dict(data, chunk_size)

    expected = [
        {'a': jnp.array([1, 2, 3, 4]), 'b': jnp.array([10, 20, 30, 40])},
        {'a': jnp.array([5, 6]), 'b': jnp.array([50, 60])},
    ]

    # Check that the length of the result matches the expected length
    assert len(result) == len(expected)

    # Check that each chunk matches the expected chunk
    for r, e in zip(result, expected):
        for k in data:
            assert_array_equal(r[k], e[k])


def test_random_sample_size():
    data = {
        'a': jnp.array([1, 2, 3, 4, 5]),
        'b': jnp.array([5, 4, 3, 2, 1]),
        'c': jnp.array([10, 20, 30, 40, 50])
    }
    num_samples = 3
    rng_key = jra.PRNGKey(123)
    sampled_data = random_sample_dict(data, num_samples, rng_key)
    for value in sampled_data.values():
        assert_(len(value) == num_samples)


def test_random_sample_consistency():
    data = {
        'a': jnp.array([1, 2, 3, 4, 5]),
        'b': jnp.array([5, 4, 3, 2, 1]),
        'c': jnp.array([10, 20, 30, 40, 50])
    }
    num_samples = 3
    rng_key = jra.PRNGKey(123)
    sampled_data1 = random_sample_dict(data, num_samples, rng_key)
    sampled_data2 = random_sample_dict(data, num_samples, rng_key)

    for key in sampled_data1:
        assert_(jnp.array_equal(sampled_data1[key], sampled_data2[key]))


def test_random_sample_difference():
    data = {
        'a': jnp.array([1, 2, 3, 4, 5]),
        'b': jnp.array([5, 4, 3, 2, 1]),
        'c': jnp.array([10, 20, 30, 40, 50])
    }
    num_samples = 3
    rng_key1, rng_key2 = jra.split(jra.PRNGKey(0))
    sampled_data1 = random_sample_dict(data, num_samples, rng_key1)
    sampled_data2 = random_sample_dict(data, num_samples, rng_key2)

    for key in sampled_data1:
        assert_(not jnp.array_equal(sampled_data1[key], sampled_data2[key]))


def test_get_keys():
    key1, key2 = get_keys()
    assert_(isinstance(key1, jnp.ndarray))
    assert_(isinstance(key2, jnp.ndarray))


def test_get_keys_different_seeds():
    key1, key2 = get_keys()
    key1a, key2a = get_keys(42)
    assert_(not onp.array_equal(key1, key1a))
    assert_(not onp.array_equal(key2, key2a))


def test_ratio_out_of_bounds():
    X = jax.random.normal(jax.random.PRNGKey(0), (100, 5))
    with pytest.raises(ValueError):
        initialize_inducing_points(X, ratio=-0.1)
    with pytest.raises(ValueError):
        initialize_inducing_points(X, ratio=1.5)


def test_invalid_method():
    X = jax.random.normal(jax.random.PRNGKey(0), (100, 5))
    with pytest.raises(ValueError):
        initialize_inducing_points(X, method='invalid_method')


def test_missing_key_for_random_method():
    X = jax.random.normal(jax.random.PRNGKey(0), (100, 5))
    with pytest.raises(ValueError):
        initialize_inducing_points(X, method='random')


@pytest.mark.parametrize("method", ["uniform", "random"])
def test_output_shape(method):
    X = jax.random.normal(jax.random.PRNGKey(0), (100, 5))
    ratio = 0.1
    inducing_points = initialize_inducing_points(
        X, ratio=ratio, method=method, key=jax.random.PRNGKey(0))
    expected_shape = (int(100 * ratio), 5)
    assert inducing_points.shape == expected_shape, "Output shape is incorrect"


@pytest.mark.skipif('sklearn' not in sys.modules, reason="sklearn is not installed")
def test_kmeans_dependency():
    X = jax.random.normal(jax.random.PRNGKey(0), (100, 5))
    try:
        inducing_points = initialize_inducing_points(X, method='kmeans')
    except ImportError:
        pytest.fail("KMeans test failed due to missing sklearn dependency")
