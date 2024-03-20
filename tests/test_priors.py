import sys
import pytest
import jax.numpy as jnp
import numpyro
from numpy.testing import assert_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.priors import place_normal_prior, place_halfnormal_prior, place_uniform_prior, place_gamma_prior, place_lognormal_prior
from gpax.priors import uniform_dist, normal_dist, halfnormal_dist, lognormal_dist, gamma_dist
from gpax.priors import auto_lognormal_priors, auto_normal_priors, auto_lognormal_kernel_priors, auto_normal_kernel_priors, auto_priors


def linear_kernel_test(X, Z, k_scale):
    # Dummy kernel functions for testing purposes
    return k_scale * jnp.dot(X, Z.T)


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
