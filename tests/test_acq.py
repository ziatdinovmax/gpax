import sys
import pytest
import numpy as onp
import jax.numpy as jnp
from numpy.testing import assert_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.gp import ExactGP
from gpax.dkl import DKL
from gpax.vidkl import viDKL
from gpax.utils import get_keys
from gpax.acquisition import EI, UCB, UE, Thompson


@pytest.mark.parametrize("acq", [EI, UCB, UE, Thompson])
def test_acq_gp(acq):
    rng_keys = get_keys()
    X = onp.random.randn(8,)
    X_new = onp.random.randn(12,)
    y = 10 * X**2
    m = ExactGP(1, 'RBF')
    m.fit(rng_keys[0], X, y, num_warmup=100, num_samples=100)
    obj = acq(rng_keys[1], m, X_new)
    assert isinstance(obj, jnp.ndarray)
    assert_equal(obj.squeeze().shape, (len(X_new),))


@pytest.mark.parametrize("acq", [EI, UCB, UE, Thompson])
def test_acq_dkl(acq):
    rng_keys = get_keys()
    X = onp.random.randn(32, 36)
    y = onp.random.randn(32,)
    X_new = onp.random.randn(10, 36)
    m = viDKL(X.shape[-1])
    m.fit(rng_keys[0], X, y, num_steps=20, step_size=0.05)
    obj = acq(rng_keys[1], m, X_new)
    assert isinstance(obj, jnp.ndarray)
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
    last_point = onp.array(X_new[3] - 0.02)
    obj2 = acq(rng_keys[1], m, X_new, last_point=last_point)
    assert_(onp.count_nonzero(obj1 - obj2) > 0)
