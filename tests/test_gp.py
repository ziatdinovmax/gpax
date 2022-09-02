import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
from numpy.testing import assert_equal, assert_array_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.gp import ExactGP
from gpax.utils import get_keys


def get_dummy_data(jax_ndarray=True, unsqueeze=False):
    X = onp.linspace(1, 2, 8) + 0.1 * onp.random.randn(8,)
    y = (10 * X**2)
    if unsqueeze:
        X = X[:, None]
    if jax_ndarray:
        return jnp.array(X), jnp.array(y)
    return X, y


def dummy_mean_fn(x, params):
    return params["a"] * x**params["b"]


def dummy_mean_fn_priors():
    a = numpyro.sample("a", numpyro.distributions.LogNormal(0, 1))
    b = numpyro.sample("b", numpyro.distributions.Normal(3, 1))
    return {"a": a, "b": b}


def gp_kernel_custom_prior():
    length = numpyro.sample("k_length", numpyro.distributions.Uniform(0, 1))
    scale = numpyro.sample("k_scale", numpyro.distributions.LogNormal(0, 1))
    return {"k_length": length, "k_scale": scale}


@pytest.mark.parametrize("jax_ndarray", [True, False])
@pytest.mark.parametrize("unsqueeze", [True, False])
@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit(kernel, jax_ndarray, unsqueeze):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray, unsqueeze)
    m = ExactGP(1, kernel)
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


@pytest.mark.parametrize("jax_ndarray", [True, False])
@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_get_samples(kernel, jax_ndarray):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray)
    m = ExactGP(1, kernel)
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100)
    samples = m.get_samples()
    assert isinstance(samples, dict)
    for k, v in samples.items():
        assert isinstance(k, str)
        assert isinstance(v, jnp.ndarray)
        assert_equal(len(v), 100)


@pytest.mark.parametrize("chain_dim, samples_dim", [(True, 2), (False, 1)])
def test_get_samples_chain_dim(chain_dim, samples_dim):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = ExactGP(1, 'RBF')
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100, num_chains=2)
    samples = m.get_samples(chain_dim)
    assert_equal(samples["k_scale"].ndim, samples_dim)
    assert_equal(samples["noise"].ndim, samples_dim)
    assert_equal(samples["k_length"].ndim, samples_dim + 1)


@pytest.mark.parametrize("kernel", ['RBF', 'Matern'])
def test_sample_kernel(kernel):
    m = ExactGP(1, kernel)
    with numpyro.handlers.seed(rng_seed=1):
        kernel_params = m._sample_kernel_params()
    _ = kernel_params.pop('period')
    param_names = ['k_length', 'k_scale']
    for k, v in kernel_params.items():
        assert k in param_names
        assert isinstance(v, jnp.ndarray)


def test_sample_periodic_kernel():
    m = ExactGP(1, 'Periodic')
    with numpyro.handlers.seed(rng_seed=1):
        kernel_params = m._sample_kernel_params()
    param_names = ['k_length', 'k_scale', 'period']
    for k, v in kernel_params.items():
        assert k in param_names
        assert isinstance(v, jnp.ndarray)


@pytest.mark.parametrize("kernel", ['RBF', 'Matern'])
def test_fit_with_custom_kernel_priors(kernel):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = ExactGP(1, kernel, kernel_prior=gp_kernel_custom_prior)
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


def test_get_mvn_posterior():
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    mean, cov = m.get_mvn_posterior(X_test, params)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(cov, jnp.ndarray)
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))


def test_get_mvn_posterior_noiseless():
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    mean1, cov1 = m.get_mvn_posterior(X_test, params, noiseless=False)
    mean1_, cov1_ = m.get_mvn_posterior(X_test, params, noiseless=False)
    mean2, cov2 = m.get_mvn_posterior(X_test, params, noiseless=True)
    assert_array_equal(mean1, mean1_)
    assert_array_equal(cov1, cov1_)
    assert_array_equal(mean1, mean2)
    assert onp.count_nonzero(cov1 - cov2) > 0


def test_single_sample_prediction():
    rng_key = get_keys()[0]
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_mean, y_sample = m._predict(rng_key, X_test, params, 1)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_sample, jnp.ndarray)
    assert_equal(y_mean.shape, X_test.squeeze().shape)
    assert_equal(y_sample.shape, (1, X_test.shape[0]))


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("unsqueeze", [True, False])
def test_prediction(unsqueeze, n):
    rng_keys = get_keys()
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=unsqueeze)
    samples = {"k_length": jax.random.normal(rng_keys[0], shape=(100, 1)),
               "k_scale": jax.random.normal(rng_keys[0], shape=(100,)),
               "noise": jax.random.normal(rng_keys[0], shape=(100,))}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_mean, y_sampled = m.predict(rng_keys[1], X_test, samples, n=n)
    assert isinstance(y_mean, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_mean.shape, X_test.squeeze().shape)
    assert_equal(y_sampled.shape, (100, n, X_test.shape[0]))


def test_noiseless_prediction():
    rng_keys = get_keys()
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    samples = {"k_length": jax.random.normal(rng_keys[0], shape=(100, 1)),
               "k_scale": jax.random.normal(rng_keys[0], shape=(100,)),
               "noise": jax.random.normal(rng_keys[0], shape=(100,))}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_mean1, y_sampled1 = m.predict(rng_keys[1], X_test, samples, n=1, noiseless=True)
    y_mean2, y_sampled2 = m.predict(rng_keys[1], X_test, samples, n=1, noiseless=False)
    assert_array_equal(y_mean1, y_mean2)
    assert onp.count_nonzero(y_sampled1 - y_sampled2) > 0


@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit_predict(kernel):
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, kernel)
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    y_pred, y_sampled = m.predict(rng_keys[1], X_test)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    print(y_sampled.shape)
    assert_equal(y_sampled.shape, (100, 1, X_test.shape[0]))


@pytest.mark.parametrize("n", [1, 10])
def test_fit_predict_in_batches(n):
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, 'RBF')
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    y_pred, y_sampled = m.predict_in_batches(rng_keys[1], X_test, batch_size=4, n=n)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    print(y_sampled.shape)
    assert_equal(y_sampled.shape, (100, n, X_test.shape[0]))


@pytest.mark.parametrize("n", [1, 10])
def test_fit_noiseless_predict_in_batches(n):
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, 'RBF')
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    y_mean1, y_sampled1 = m.predict_in_batches(rng_keys[1], X_test, batch_size=4, n=n, noiseless=True)
    y_mean2, y_sampled2 = m.predict_in_batches(rng_keys[1], X_test, batch_size=4, n=n, noiseless=False)
    assert_array_equal(y_mean1, y_mean2)
    assert onp.count_nonzero(y_sampled1 - y_sampled2) > 0


@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_fit_with_mean_fn(jax_ndarray):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray)
    m = ExactGP(1, 'RBF', mean_fn = lambda x: 8*x**2)
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_fit_with_prob_mean_fn(jax_ndarray):
    rng_key = get_keys()[0]
    X, y = get_dummy_data(jax_ndarray)
    m = ExactGP(1, 'RBF', mean_fn=dummy_mean_fn, mean_fn_prior=dummy_mean_fn_priors)
    m.fit(rng_key, X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


def test_fit_predict_with_mean_fn():
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, 'RBF', mean_fn = lambda x: 8*x**2)
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    y_pred, y_sampled = m.predict(rng_keys[1], X_test)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    print(y_sampled.shape)
    assert_equal(y_sampled.shape, (100, 1, X_test.shape[0]))


def test_fit_predict_with_prob_mean_fn():
    rng_keys = get_keys()
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, 'RBF', mean_fn=dummy_mean_fn, mean_fn_prior=dummy_mean_fn_priors)
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    y_pred, y_sampled = m.predict(rng_keys[1], X_test)
    assert isinstance(y_pred, jnp.ndarray)
    assert isinstance(y_sampled, jnp.ndarray)
    assert_equal(y_pred.shape, X_test.squeeze().shape)
    assert_equal(y_sampled.shape, (100, 1, X_test.shape[0]))


def test_sample_from_prior():
    rng_key, _ = get_keys()
    X, _ = get_dummy_data()
    m = ExactGP(1, 'RBF')
    prior_pred = m.sample_from_prior(rng_key, X, num_samples=8)
    assert_equal(prior_pred.shape, (8, X.shape[0]))


def test_jitter_fit():
    rng_key, _ = get_keys()
    X, y = get_dummy_data()
    m = ExactGP(1, 'RBF')
    m.fit(rng_key, X, y, num_samples=50, num_warmup=50, jitter=1e-6)
    samples1 = m.get_samples()
    m = ExactGP(1, 'RBF')
    m.fit(rng_key, X, y, num_samples=50, num_warmup=50, jitter=1e-6)
    samples1a = m.get_samples()
    m = ExactGP(1, 'RBF')
    m.fit(rng_key, X, y, num_samples=50, num_warmup=50, jitter=1e-5)
    samples2 = m.get_samples()
    assert_(onp.count_nonzero(samples1["k_length"] - samples1a["k_length"]) == 0)
    assert_(onp.count_nonzero(samples1["k_length"] - samples2["k_length"]) > 0)


def test_jitter_predict():
    rng_keys = get_keys()
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    samples = {"k_length": jax.random.normal(rng_keys[0], shape=(100, 1)),
               "k_scale": jax.random.normal(rng_keys[0], shape=(100,)),
               "noise": jax.random.normal(rng_keys[0], shape=(100,))}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_mean1, y_sampled1 = m.predict(rng_keys[1], X_test, samples, n=1, jitter=1e-6)
    y_mean2, y_sampled2 = m.predict(rng_keys[1], X_test, samples, n=1, jitter=1e-5)
    assert_(onp.count_nonzero(y_sampled1 - y_sampled2) > 0)
    assert_(onp.count_nonzero(y_mean1 - y_mean2) > 0)
