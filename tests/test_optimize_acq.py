import sys
import pytest
import numpy as onp
import jax.numpy as jnp
from numpy.testing import assert_

sys.path.insert(0, "../gpax/")

from gpax.models.gp import ExactGP
from gpax.acquisition.optimize import optimize_acq
from gpax.acquisition.acquisition import UCB, EI
from gpax.utils import get_keys


def get_inputs():
    X = onp.random.uniform(-2, 2, size=(4,))
    y = X**3
    return X, y


@pytest.mark.parametrize("acq_fn", [UCB, EI])
def test_optimize_acq(acq_fn):
    lower_bound = -2.0
    upper_bound = 2.0
    num_initial_guesses = 3
    key1, key2 = get_keys()
    X, y = get_inputs()
    model = ExactGP(1, 'RBF')
    model.fit(key1, X, y, num_warmup=50, num_samples=50)
    x_next = optimize_acq(
        key2, model, acq_fn, num_initial_guesses, lower_bound, upper_bound)
    assert_(isinstance(x_next, jnp.ndarray))
    


    