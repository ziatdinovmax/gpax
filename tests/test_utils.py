import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax.random as jra
import numpyro
from numpy.testing import assert_equal, assert_, assert_array_equal

sys.path.insert(0, "../gpax/")

from gpax.utils import preprocess_sparse_image, split_dict, random_sample_dict, get_keys
from gpax.utils import place_normal_prior, place_halfnormal_prior, place_uniform_prior, place_gamma_prior, gamma_dist, uniform_dist, normal_dist, halfnormal_dist


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


@pytest.mark.parametrize("prior", [place_normal_prior, place_halfnormal_prior])
def test_normal_prior(prior):
    with numpyro.handlers.seed(rng_seed=1):
        sample = prior("a")
    assert_(isinstance(sample, jnp.ndarray))


def test_uniform_prior():
    with numpyro.handlers.seed(rng_seed=1):
        sample = place_uniform_prior("a", 0, 1)
    assert_(isinstance(sample, jnp.ndarray))


def test_gamma_prior():
    with numpyro.handlers.seed(rng_seed=1):
        sample = place_gamma_prior("a", 2, 2)
    assert_(isinstance(sample, jnp.ndarray))


def test_normal_prior_params():
    with numpyro.handlers.seed(rng_seed=1):
        with numpyro.handlers.trace() as tr:
            place_normal_prior("a", loc=0.5, scale=0.1)
    site = tr["a"]
    assert_(isinstance(site['fn'], numpyro.distributions.Normal))
    assert_equal(site['fn'].loc, 0.5)
    assert_equal(site['fn'].scale, 0.1)


def test_halfnormal_prior_params():
    with numpyro.handlers.seed(rng_seed=1):
        with numpyro.handlers.trace() as tr:
            place_halfnormal_prior("a", 0.1)
    site = tr["a"]
    assert_(isinstance(site['fn'], numpyro.distributions.HalfNormal))
    assert_equal(site['fn'].scale, 0.1)


def test_uniform_prior_params():
    with numpyro.handlers.seed(rng_seed=1):
        with numpyro.handlers.trace() as tr:
            place_uniform_prior("a", low=0.5, high=1.0)
    site = tr["a"]
    assert_(isinstance(site['fn'], numpyro.distributions.Uniform))
    assert_equal(site['fn'].low, 0.5)
    assert_equal(site['fn'].high, 1.0)


def test_gamma_prior_params():
    with numpyro.handlers.seed(rng_seed=1):
        with numpyro.handlers.trace() as tr:
            place_gamma_prior("a", c=2.0, r=1.0)
    site = tr["a"]
    assert_(isinstance(site['fn'], numpyro.distributions.Gamma))
    assert_equal(site['fn'].concentration, 2.0)
    assert_equal(site['fn'].rate, 1.0)


def test_get_uniform_dist():
    uniform_dist_ = uniform_dist(low=1.0, high=5.0)
    assert isinstance(uniform_dist_, numpyro.distributions.Uniform)
    assert uniform_dist_.low == 1.0
    assert uniform_dist_.high == 5.0


def test_get_uniform_dist_infer_params():
    uniform_dist_ = uniform_dist(input_vec=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert uniform_dist_.low == 1.0
    assert uniform_dist_.high == 5.0


def test_get_gamma_dist():
    gamma_dist_ = gamma_dist(c=2.0, r=1.0)
    assert isinstance(gamma_dist_, numpyro.distributions.Gamma)
    assert gamma_dist_.concentration == 2.0
    assert gamma_dist_.rate == 1.0


def test_get_normal_dist():
    normal_dist_ = normal_dist(loc=2.0, scale=3.0)
    assert isinstance(normal_dist_, numpyro.distributions.Normal)
    assert normal_dist_.loc == 2.0
    assert normal_dist_.scale == 3.0


def test_get_halfnormal_dist():
    halfnormal_dist_ = halfnormal_dist(scale=1.5)
    assert isinstance(halfnormal_dist_, numpyro.distributions.HalfNormal)
    assert halfnormal_dist_.scale == 1.5


def test_get_gamma_dist_infer_param():
    gamma_dist_ = gamma_dist(input_vec=jnp.linspace(0, 10, 20))
    assert isinstance(gamma_dist_, numpyro.distributions.Gamma)
    assert gamma_dist_.concentration == 5.0
    assert gamma_dist_.rate == 1.0


def test_get_uniform_dist_error():
    with pytest.raises(ValueError):
        uniform_dist(low=1.0)  # Only low provided without input_vec
    with pytest.raises(ValueError):
        uniform_dist(high=5.0)  # Only high provided without input_vec
    with pytest.raises(ValueError):
        uniform_dist()  # Neither low nor high, and no input_vec


def test_get_gamma_dist_error():
    with pytest.raises(ValueError):
        uniform_dist()  # Neither concentration, nor input_vec



