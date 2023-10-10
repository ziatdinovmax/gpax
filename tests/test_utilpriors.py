import sys
import pytest
import jax.numpy as jnp
import numpyro
from numpy.testing import assert_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.utils import place_normal_prior, place_halfnormal_prior, place_uniform_prior, place_gamma_prior, place_lognormal_prior
from gpax.utils import uniform_dist, normal_dist, halfnormal_dist, lognormal_dist, gamma_dist
from gpax.utils import auto_lognormal_priors, auto_normal_priors, auto_lognormal_kernel_priors, auto_normal_kernel_priors, auto_priors
from gpax.utils import set_fn, set_kernel_fn, _set_noise_kernel_fn


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


@pytest.mark.parametrize("prior", [place_normal_prior, place_halfnormal_prior, place_lognormal_prior])
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


def test_lognormal_prior_params():
    with numpyro.handlers.seed(rng_seed=1):
        with numpyro.handlers.trace() as tr:
            place_lognormal_prior("a", loc=0.5, scale=0.1)
    site = tr["a"]
    assert_(isinstance(site['fn'], numpyro.distributions.LogNormal))
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


def test_get_lognormal_dist():
    lognormal_dist_ = lognormal_dist(loc=2.0, scale=3.0)
    assert isinstance(lognormal_dist_, numpyro.distributions.LogNormal)
    assert lognormal_dist_.loc == 2.0
    assert lognormal_dist_.scale == 3.0


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


@pytest.mark.parametrize("prior_type", ["normal", "lognormal"])
def test_auto_priors(prior_type):
    prior_fn = auto_priors(sample_function, 1, prior_type, loc=2.0, scale=1.0)
    with numpyro.handlers.seed(rng_seed=1):
        with numpyro.handlers.trace() as tr:
            prior_fn()
    site1 = tr["a"]
    assert_(isinstance(site1['fn'], numpyro.distributions.Distribution))
    assert_equal(site1['fn'].loc, 2.0)
    assert_equal(site1['fn'].scale, 1.0)
    site2 = tr["b"]
    assert_(isinstance(site2['fn'], numpyro.distributions.Distribution))
    assert_equal(site2['fn'].loc, 2.0)
    assert_equal(site2['fn'].scale, 1.0)


@pytest.mark.parametrize("autopriors", [auto_normal_priors, auto_lognormal_priors])
def test_auto_normal_priors(autopriors):
    priors_fn = autopriors(sample_function)
    with numpyro.handlers.seed(rng_seed=1):
        with numpyro.handlers.trace() as tr:
            priors_fn()
    assert_('a' in tr)
    assert_('b' in tr)


@pytest.mark.parametrize("autopriors", [auto_normal_kernel_priors, auto_lognormal_kernel_priors])
def test_auto_normal_kernel_priors(autopriors):
    priors_fn = autopriors(linear_kernel_test)
    with numpyro.handlers.seed(rng_seed=1):
        with numpyro.handlers.trace() as tr:
            priors_fn()
    assert_('k_scale' in tr)


def test_set_noise_kernel_fn():
    from gpax.kernels import RBFKernel

    X = jnp.array([[1, 2], [3, 4], [5, 6]])
    params_i = {"k_length": jnp.array([1.0]), "k_scale": jnp.array(1.0)}
    params = {"k_noise_length": jnp.array([1.0]), "k_noise_scale": jnp.array(1.0)}
    noise_rbf = _set_noise_kernel_fn(RBFKernel)
    assert_(jnp.array_equal(noise_rbf(X, X, params), RBFKernel(X, X, params_i)))
