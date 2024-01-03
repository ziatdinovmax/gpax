import sys
import pytest
import numpy as onp
import jax
import jax.numpy as jnp
from numpy.testing import assert_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.models.gp import ExactGP
from gpax.models.vidkl import viDKL
from gpax.models import DKL
from gpax.utils import get_keys
from gpax.acquisition.base_acq import ei, ucb, poi, ue, kg
from gpax.acquisition.acquisition import _compute_mean_and_var
from gpax.acquisition.acquisition import EI, UCB, UE, POI, Thompson, KG
from gpax.acquisition.batch_acquisition import _compute_batch_acquisition
from gpax.acquisition.batch_acquisition import qEI, qPOI, qUCB, qKG
from gpax.acquisition.penalties import compute_penalty


class mock_GP:
    def __init__(self):
        self.mcmc = 1

    def get_samples(self):
        rng_key = get_keys()[1]
        samples = {"k_length": jax.random.normal(rng_key, shape=(100, 1)),
                   "k_scale": jax.random.normal(rng_key, shape=(100,)),
                   "noise": jax.random.normal(rng_key, shape=(100,))}
        return samples


@pytest.mark.parametrize("base_acq", [ei, ucb, poi, ue])
def test_base_standard_acq(base_acq):
    mean = onp.random.randn(10,)
    var = onp.random.uniform(0, 1, size=10)
    moments = (mean, var)
    obj = base_acq(moments)
    assert_(isinstance(obj, jnp.ndarray))
    assert_equal(len(obj), len(mean))
    assert_equal(obj.ndim, 1)


def test_base_acq_kg():
    rng_keys = get_keys()
    X = onp.random.randn(8,)
    X_new = onp.random.randn(12, 1)
    y = 10 * X**2
    m = ExactGP(1, 'RBF')
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    sample = {k: v[0] for (k, v) in m.get_samples().items()}
    obj = kg(m, X_new, sample)
    assert_(isinstance(obj, jnp.ndarray))
    assert_equal(len(obj), len(X_new))
    assert_equal(obj.ndim, 1)


@pytest.mark.parametrize("base_acq", [ei, ucb, poi])
def test_base_standard_acq_maximize(base_acq):
    mean = onp.random.randn(10,)
    var = onp.random.uniform(0, 1, size=10)
    moments = (mean, var)
    obj1 = base_acq(moments, maximize=False)
    obj2 = base_acq(moments, maximize=True)
    assert_(not onp.array_equal(obj1, obj2))


@pytest.mark.parametrize("base_acq", [ei, poi])
def test_base_standard_acq_best_f(base_acq):
    mean = onp.random.randn(10,)
    var = onp.random.uniform(0, 1, size=10)
    best_f = mean.min() - 0.01
    moments = (mean, var)
    obj1 = base_acq(moments)
    obj2 = base_acq(moments, best_f=best_f)
    assert_(not onp.array_equal(obj1, obj2))


def test_compute_mean_and_var():
    rng_keys = get_keys()
    X = onp.random.randn(8,)
    X_new = onp.random.randn(12,)
    y = 10 * X**2
    m = ExactGP(1, 'RBF')
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    mean, var = _compute_mean_and_var(
        rng_keys[1], m, X_new, n=1, noiseless=True)
    assert_equal(mean.shape, (len(X_new),))
    assert_equal(var.shape, (len(X_new),))


@pytest.mark.parametrize("acq", [EI, UCB, UE, Thompson, POI])
def test_acq_gp(acq):
    rng_keys = get_keys()
    X = onp.random.randn(8,)
    X_new = onp.random.randn(12,)
    y = 10 * X**2
    m = ExactGP(1, 'RBF')
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    obj = acq(rng_keys[1], m, X_new)
    assert_(isinstance(obj, jnp.ndarray))
    assert_equal(obj.squeeze().shape, (len(X_new),))


@pytest.mark.parametrize("acq", [EI, UCB, UE, Thompson, POI, KG])
def test_acq_vidkl(acq):
    rng_keys = get_keys()
    X = onp.random.randn(8, 10)
    X_new = onp.random.randn(12, 10)
    y = (10 * X**2).mean(-1)
    m = viDKL(X.shape[-1], 2, 'RBF')
    m.fit(rng_keys[0], X, y, num_steps=10)
    obj = acq(rng_keys[1], m, X_new)
    assert_(isinstance(obj, jnp.ndarray))
    assert_equal(obj.squeeze().shape, (len(X_new),))


@pytest.mark.parametrize("acq", [EI, POI, UCB])
def test_acq_dkl(acq):
    rng_keys = get_keys()
    X = onp.random.randn(12, 8)
    y = onp.random.randn(12,)
    X_new = onp.random.randn(10, 8)[None]
    m = DKL(X.shape[-1], 2, 'RBF')
    m.fit(rng_keys[0], X, y, num_samples=5, num_warmup=5)
    obj = acq(rng_keys[1], m, X_new, subsample_size=4)
    assert_equal(obj.shape, (X_new.shape[1],))


def test_UCB_beta():
    rng_keys = get_keys()
    X = onp.random.randn(8,)
    X_new = onp.random.randn(12,)
    y = 10 * X**2
    m = ExactGP(1, 'RBF')
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    obj1 = UCB(rng_keys[1], m, X_new, beta=2)
    obj2 = UCB(rng_keys[1], m, X_new, beta=4)
    obj3 = UCB(rng_keys[1], m, X_new, beta=2)
    assert_(not onp.array_equal(obj1, obj2))
    assert_(onp.array_equal(obj1, obj3))


def test_EI_gp_penalty_inv_distance():
    rng_keys = get_keys()
    X = onp.random.randn(8,)
    X_new = onp.random.randn(12,)
    recent_points = X_new[-3:-1]
    y = 10 * X**2
    m = ExactGP(1, 'RBF')
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    obj1 = EI(rng_keys[1], m, X_new)
    obj2 = EI(rng_keys[1], m, X_new, penalty="inverse_distance", recent_points=recent_points)
    assert_(obj2[-1] < obj1[-1])
    assert_(obj2[-2] < obj1[-2])


def test_UCB_gp_penalty_inv_distance():
    rng_keys = get_keys()
    X = onp.random.randn(8,)
    X_new = onp.random.randn(12,)
    recent_points = X_new[-3:-1]
    y = 10 * X**2
    m = ExactGP(1, 'RBF')
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    obj1 = UCB(rng_keys[1], m, X_new)
    obj2 = UCB(rng_keys[1], m, X_new, penalty="inverse_distance", recent_points=recent_points)
    assert_(obj2[-1] < obj1[-1])
    assert_(obj2[-2] < obj1[-2])


def test_UE_gp_penalty_inv_distance():
    rng_keys = get_keys()
    X = onp.random.randn(8,)
    X_new = onp.random.randn(12,)
    recent_points = X_new[-3:-1]
    y = 10 * X**2
    m = ExactGP(1, 'RBF')
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    obj1 = UE(rng_keys[1], m, X_new)
    obj2 = UE(rng_keys[1], m, X_new, penalty="inverse_distance", recent_points=recent_points)
    assert_(obj2[-1] < obj1[-1])
    assert_(obj2[-2] < obj1[-2])


@pytest.mark.parametrize("maximize_distance", [False, True])
def test_compute_batch_acquisition(maximize_distance):
    def mock_acq_fn(*args):
        return jnp.arange(0, 10)
    X = onp.random.randn(10)
    rng_key = get_keys()[0]
    m = mock_GP()
    obj = _compute_batch_acquisition(
        rng_key, m, X, mock_acq_fn, subsample_size=7,
        maximize_distance=maximize_distance)
    assert_equal(obj.shape[0], 7)


@pytest.mark.parametrize("q", [1, 3])
@pytest.mark.parametrize("acq", [qEI, qPOI, qUCB, qKG])
def test_batched_acq_gp(acq, q):
    rng_key = get_keys()
    X = onp.random.randn(8,)
    X_new = onp.random.randn(12,)
    y = 10 * X**2
    m = ExactGP(1, 'RBF')
    m.fit(rng_key[0], X, y, num_warmup=100, num_samples=100)
    obj = acq(rng_key[1], m, X_new, subsample_size=q)
    assert_equal(obj.shape, (q, len(X_new)))


@pytest.mark.parametrize('pen', ['delta', 'inverse_distance'])
@pytest.mark.parametrize("acq", [EI, UCB, UE])
def test_acq_penalty_indices(acq, pen):
    rng_keys = get_keys()
    h = w = 5
    X = onp.random.randn(h*w, 16)
    y = onp.random.randn(len(X))
    indices = onp.array([(i, j) for i in range(h) for j in range(w)])
    X_new = onp.random.randn(h*w, 16)
    m = viDKL(input_dim=16)
    m.fit(rng_keys[0], X, y, num_steps=50)
    obj1 = acq(rng_keys[1], m, X_new, grid_indices=indices,
               penalty=pen, recent_points=indices[5:7])
    obj2 = acq(rng_keys[1], m, X_new)
    assert_(isinstance(obj1, jnp.ndarray))
    assert_equal(obj1.squeeze().shape, (len(X_new),))
    assert_(onp.count_nonzero(obj1 - obj2) > 0)


def test_compute_penalty_delta():
    X = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    recent_points = jnp.array([[4, 5, 6], [1, 2, 3]])
    penalty_factor = 1.0
    penalties = compute_penalty(X, recent_points, "delta", penalty_factor)
    assert jnp.array_equal(penalties, jnp.array([jnp.inf, jnp.inf, 0.0]))


def test_compute_penalty_inverse_distance():
    X = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    recent_points = jnp.array([[4, 5, 6], [7, 8, 9]])
    penalty_factor = 1.0
    penalties = compute_penalty(X, recent_points, "inverse_distance", penalty_factor)
    assert_(penalties[-1] > penalties[-2])
    assert_(penalties[-2] > penalties[-3])


@pytest.mark.parametrize("acq_func", [EI, UCB, UE])
def test_acq_error(acq_func):
    # Initialize the required inputs
    rng_keys = get_keys()
    X = onp.random.randn(8,)
    X_new = onp.random.randn(12,)
    recent_points = X_new[-3:-1]
    y = 10 * X**2
    m = ExactGP(1, 'RBF')
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)

    # Test the ValueError for not passing the recent_points when penalty is not None
    with pytest.raises(ValueError):
        acq_func(rng_keys[1], m, X_new, penalty='delta')

    # Test the ValueError for passing recent_points as non ndarray type
    with pytest.raises(ValueError):
        acq_func(rng_keys[1], m, X_new, penalty='delta', recent_points=[1, 2, 3])

    # Test the function with correct parameters
    try:
        output = acq_func(rng_keys[1], m, X, penalty='delta', recent_points=recent_points)
        assert isinstance(output, jnp.ndarray)  # check if output is of correct type
    except Exception as e:
        pytest.fail(f"Test failed, error: {str(e)}")
