import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
from numpy.testing import assert_equal, assert_array_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.models.gp import ExactGP
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


@pytest.fixture(scope="module", params=['RBF', 'Matern', 'Periodic'])
def fitted_model(request):
    kernel = request.param
    X, y = get_dummy_data()
    m = ExactGP(1, kernel)
    m.fit(X, y, num_warmup=10, num_samples=10)
    return m


@pytest.fixture(scope="module")
def fitted_model_rbf():
    kernel = 'RBF'
    X, y = get_dummy_data()
    m = ExactGP(1, kernel)
    m.fit(X, y, num_warmup=10, num_samples=10)
    return m


@pytest.mark.parametrize("jax_ndarray", [True, False])
@pytest.mark.parametrize("unsqueeze", [True, False])
@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_fit(kernel, jax_ndarray, unsqueeze):
    X, y = get_dummy_data(jax_ndarray, unsqueeze)
    m = ExactGP(1, kernel)
    m.fit(X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


@pytest.mark.parametrize("jax_ndarray", [True, False])
@pytest.mark.parametrize("kernel", ['RBF', 'Matern', 'Periodic'])
def test_get_samples(kernel, jax_ndarray):
    X, y = get_dummy_data(jax_ndarray)
    m = ExactGP(1, kernel)
    m.fit(X, y, num_warmup=100, num_samples=100)
    samples = m.get_samples()
    assert isinstance(samples, dict)
    for k, v in samples.items():
        assert isinstance(k, str)
        assert isinstance(v, jnp.ndarray)
        assert_equal(len(v), 100)


@pytest.mark.parametrize("chain_dim, samples_dim", [(True, 2), (False, 1)])
def test_get_samples_chain_dim(fitted_model_rbf, chain_dim, samples_dim):
    m = fitted_model_rbf
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


def test_sample_noise():
    m = ExactGP(1, 'RBF')
    with numpyro.handlers.seed(rng_seed=1):
        noise = m._sample_noise()
    assert isinstance(noise, jnp.ndarray)


def test_sample_noise_custom_prior():
    noise_prior_dist = numpyro.distributions.HalfNormal(.1)
    m1 = ExactGP(1, 'RBF')
    with numpyro.handlers.seed(rng_seed=1):
        noise1 = m1._sample_noise()
    m2 = ExactGP(1, 'RBF', noise_prior_dist=noise_prior_dist)
    with numpyro.handlers.seed(rng_seed=1):
        noise2 = m2._sample_noise()
    assert_(not onp.array_equal(noise1, noise2))


def test_sample_kernel_custom_lscale_prior():
    lscale_prior_dist = numpyro.distributions.Normal(20, .1)
    m1 = ExactGP(1, 'RBF')
    with numpyro.handlers.seed(rng_seed=1):
        lscale1 = m1._sample_kernel_params()["k_length"]
    m2 = ExactGP(1, 'RBF', lengthscale_prior_dist=lscale_prior_dist)
    with numpyro.handlers.seed(rng_seed=1):
        lscale2 = m2._sample_kernel_params()["k_length"]
    assert_(not onp.array_equal(lscale1, lscale2))


@pytest.mark.parametrize("kernel", ['RBF', 'Matern'])
def test_fit_with_custom_kernel_priors(kernel):
    X, y = get_dummy_data()
    m = ExactGP(1, kernel, kernel_prior=gp_kernel_custom_prior)
    m.fit(X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


def test_compute_gp_posterior():
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = ExactGP(1, 'RBF')
    mean, cov = m.compute_gp_posterior(X_test, X, y, params)
    assert isinstance(mean, jnp.ndarray)
    assert isinstance(cov, jnp.ndarray)
    assert_equal(mean.shape, (X_test.shape[0],))
    assert_equal(cov.shape, (X_test.shape[0], X_test.shape[0]))


def test_compute_gp_posterior_noiseless():
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = ExactGP(1, 'RBF')
    mean1, cov1 = m.compute_gp_posterior(X_test, X, y, params, noiseless=False)
    mean1_, cov1_ = m.compute_gp_posterior(X_test, X, y, params, noiseless=False)
    mean2, cov2 = m.compute_gp_posterior(X_test, X, y, params, noiseless=True)
    assert_array_equal(mean1, mean1_)
    assert_array_equal(cov1, cov1_)
    assert_array_equal(mean1, mean2)
    assert onp.count_nonzero(cov1 - cov2) > 0


def test_draw_from_mvn():
    rng_key = get_keys()[0]
    X, y = get_dummy_data(unsqueeze=True)
    X_test, _ = get_dummy_data(unsqueeze=True)
    params = {"k_length": jnp.array([1.0]),
              "k_scale": jnp.array(1.0),
              "noise": jnp.array(0.1)}
    m = ExactGP(1, 'RBF')
    m.X_train = X
    m.y_train = y
    y_sample = m.draw_from_mvn(rng_key, X_test, params, 1, noiseless=True)
    assert isinstance(y_sample, jnp.ndarray)
    assert_equal(y_sample.shape, (1, X_test.shape[0]))


def test_noiseless_prediction(fitted_model_rbf):
    m = fitted_model_rbf
    X_test, _ = get_dummy_data(unsqueeze=True)
    p_mean1, p_var1 = m.predict(X_test, noiseless=True)
    p_mean2, p_var2 = m.predict(X_test, noiseless=False)
    assert_array_equal(p_mean1, p_mean2)
    assert onp.count_nonzero(p_var1 - p_var2) > 0


def test_fit_predict(fitted_model):
    m = fitted_model
    X_test, _ = get_dummy_data()
    p_mean, p_var = m.predict(X_test)
    assert isinstance(p_mean, jnp.ndarray)
    assert isinstance(p_var, jnp.ndarray)
    assert_equal(p_mean.shape, X_test.shape)
    assert_equal(p_var.shape, p_mean.shape)


@pytest.mark.parametrize("batch_size", [2, 6])
def test_fit_predict_in_batches(fitted_model_rbf, batch_size):
    m = fitted_model_rbf
    X_test, _ = get_dummy_data()
    p_mean, p_var = m.predict_in_batches(X_test, batch_size=batch_size)
    assert isinstance(p_mean, jnp.ndarray)
    assert isinstance(p_var, jnp.ndarray)
    assert_equal(p_mean.shape, X_test.shape)
    assert_equal(p_var.shape, p_mean.shape)


def test_fit_noiseless_predict_in_batches(fitted_model_rbf):
    m = fitted_model_rbf
    X_test, _ = get_dummy_data()
    p_mean1, p_var1 = m.predict_in_batches(X_test, batch_size=4, noiseless=True)
    p_mean2, p_var2 = m.predict_in_batches(X_test, batch_size=4, noiseless=False)
    assert_array_equal(p_mean1, p_mean2)
    assert onp.count_nonzero(p_var1 - p_var2) > 0


@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_fit_with_mean_fn(jax_ndarray):
    X, y = get_dummy_data(jax_ndarray)
    m = ExactGP(1, 'RBF', mean_fn = lambda x: 8*x**2)
    m.fit(X, y, num_warmup=10, num_samples=10)
    assert m.mcmc is not None


@pytest.mark.parametrize("jax_ndarray", [True, False])
def test_fit_with_prob_mean_fn(jax_ndarray):
    X, y = get_dummy_data(jax_ndarray)
    m = ExactGP(1, 'RBF', mean_fn=dummy_mean_fn, mean_fn_prior=dummy_mean_fn_priors)
    m.fit(X, y, num_warmup=100, num_samples=100)
    assert m.mcmc is not None


def test_fit_predict_with_mean_fn():
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, 'RBF', mean_fn = lambda x: 8*x**2)
    m.fit(X, y, num_warmup=100, num_samples=100)
    p_mean, p_var = m.predict(X_test)
    assert isinstance(p_mean, jnp.ndarray)
    assert isinstance(p_var, jnp.ndarray)
    assert_equal(p_mean.shape, X_test.shape)
    assert_equal(p_var.shape, p_mean.shape)


def test_fit_predict_with_prob_mean_fn():
    X, y = get_dummy_data()
    X_test, _ = get_dummy_data()
    m = ExactGP(1, 'RBF', mean_fn=dummy_mean_fn, mean_fn_prior=dummy_mean_fn_priors)
    m.fit(X, y, num_warmup=10, num_samples=10)
    p_mean, p_var = m.predict(X_test)
    assert isinstance(p_mean, jnp.ndarray)
    assert isinstance(p_var, jnp.ndarray)
    assert_equal(p_mean.shape, X_test.shape)
    assert_equal(p_var.shape, p_mean.shape)


def test_sample_from_prior():
    X, _ = get_dummy_data()
    m = ExactGP(1, 'RBF')
    prior_pred = m.sample_from_prior(X, num_samples=8)
    assert_equal(prior_pred.shape, (8, X.shape[0]))


def test_jitter_fit():
    X, y = get_dummy_data()
    m = ExactGP(1, 'RBF', jitter=1e-6)
    m.fit(X, y, num_samples=50, num_warmup=50)
    samples1 = m.get_samples()
    m = ExactGP(1, 'RBF', jitter=1e-6)
    m.fit(X, y, num_samples=50, num_warmup=50)
    samples1a = m.get_samples()
    m = ExactGP(1, 'RBF', jitter=1e-5)
    m.fit(X, y, num_samples=50, num_warmup=50)
    samples2 = m.get_samples()
    assert_(onp.count_nonzero(samples1["k_length"] - samples1a["k_length"]) == 0)
    assert_(onp.count_nonzero(samples1["k_length"] - samples2["k_length"]) > 0)
