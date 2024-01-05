import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist
from numpy.testing import assert_equal, assert_array_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.models.uigp import UIGP
from gpax.utils import get_keys


def get_dummy_data():
    X = onp.linspace(1, 2, 8) + 0.1 * onp.random.randn(8,)
    X_prime = onp.random.normal(X, 0.1)
    y = (10 * X_prime**2)
    return jnp.array(X_prime), jnp.array(y)


def test_fit():
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    m = UIGP(1, 'RBF')
    m.fit(rng_key, X, y, num_warmup=10, num_samples=10)
    assert m.mcmc is not None