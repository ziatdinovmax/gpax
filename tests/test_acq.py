import sys
import pytest
import numpy as onp
import jax.numpy as jnp
from numpy.testing import assert_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.models.gp import ExactGP
from gpax.models.vidkl import viDKL
from gpax.utils import get_keys
from gpax.acquisition import EI, UCB, UE, Thompson
from gpax.acquisition.penalties import compute_penalty, penalty_point, find_and_replace_point_indices


@pytest.mark.parametrize("acq", [EI, UCB, UE, Thompson])
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


@pytest.mark.parametrize("acq", [EI, UCB, UE, Thompson])
def test_acq_dkl(acq):
    rng_keys = get_keys()
    X = onp.random.randn(32, 36)
    y = onp.random.randn(32,)
    X_new = onp.random.randn(10, 36)
    m = viDKL(X.shape[-1])
    m.fit(rng_keys[0], X, y, num_steps=20, step_size=0.05)
    obj = acq(rng_keys[1], m, X_new)
    assert_(isinstance(obj, jnp.ndarray))
    assert_equal(obj.squeeze().shape, (len(X_new),))


@pytest.mark.parametrize("acq", [EI, UCB, UE])
def test_acq_penalty(acq):
    rng_keys = get_keys()
    X = onp.random.randn(8,)
    X_new = onp.random.randn(12,)
    y = 10 * X**2
    m = ExactGP(1, 'RBF')
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    obj1 = acq(rng_keys[1], m, X_new)
    recent_points = onp.array(X_new[2:4] - 0.02)
    obj2 = acq(rng_keys[1], m, X_new, distance_penalty=0.1, recent_points=recent_points)
    assert_(onp.count_nonzero(obj1 - obj2) > 0)


@pytest.mark.parametrize("acq", [EI, UCB, UE])
def test_acq_penalty_indices(acq):
    rng_keys = get_keys()
    h = w = 5
    X = onp.random.randn(h*w, 16)
    y = onp.random.randn(len(X))
    indices = onp.array([(i, j) for i in range(h) for j in range(w)])
    X_new = onp.random.randn(h*w, 16)
    m = viDKL(input_dim=16)
    m.fit(rng_keys[0], X, y, num_steps=50)
    obj1 = acq(rng_keys[1], m, X_new, grid_indices=indices,
               distance_penalty=0.1, recent_points=indices[5:7])
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


def test_compute_penalty_delta_no_recent_points():
    X = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    penalty_factor = 1.0
    penalties = compute_penalty(X, None, "delta", penalty_factor)
    assert jnp.array_equal(penalties, jnp.array([0.0, 0.0, jnp.inf]))


def test_compute_penalty_inverse_distance():
    X = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    recent_points = jnp.array([[4, 5, 6], [7, 8, 9]])
    penalty_factor = 1.0
    penalties = compute_penalty(X, recent_points, "inverse_distance", penalty_factor)
    assert_(penalties[-1] > penalties[-2])
    assert_(penalties[-2] > penalties[-3])


def test_compute_penalty_inverse_distance_no_recent_points():
    X = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    penalty_factor = 1.0
    penalties = compute_penalty(X, None, "inverse_distance", penalty_factor)
    assert_(isinstance(penalties, jnp.ndarray))
    assert_(penalties.shape == (3,))