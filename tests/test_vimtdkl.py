import sys
import pytest
import numpy as onp
import jax.numpy as jnp
from numpy.testing import assert_, assert_equal

sys.path.insert(0, "../gpax/")

from gpax.models.vi_mtdkl import viMTDKL
from gpax.utils import get_keys


def get_dummy_data():
    X = onp.random.randn(21, 36)
    y = onp.random.randn(21,)
    return jnp.array(X), jnp.array(y)


def attach_indices(X, num_tasks):
    indices = onp.random.randint(0, num_tasks, size=len(X))
    return jnp.column_stack([X, indices])


@pytest.mark.parametrize("num_latents", [1, 2])
@pytest.mark.parametrize("num_tasks", [2, 3])
@pytest.mark.parametrize("data_kernel", ['RBF', 'Matern'])
def test_fit_multitask(data_kernel, num_tasks, num_latents):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X = attach_indices(X, num_tasks)
    m = viMTDKL(X.shape[-1] - 1, 2, data_kernel, num_latents=num_latents, shared_input_space=False)
    m.fit(rng_key, X, y, num_steps=10)
    assert_(isinstance(m.kernel_params, dict))
    assert_(isinstance(m.nn_params, dict))


@pytest.mark.parametrize("num_latents", [1, 2])
@pytest.mark.parametrize("num_tasks", [2, 3])
@pytest.mark.parametrize("data_kernel", ['RBF', 'Matern'])
def test_fit_multitask_shared_input(data_kernel, num_tasks, num_latents):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    y = jnp.repeat(y[:, None], num_tasks, axis=1).reshape(-1)
    m = viMTDKL(X.shape[-1], 2, data_kernel, num_latents=num_latents,
                shared_input_space=True, num_tasks=num_tasks)
    m.fit(rng_key, X, y, num_steps=10)
    assert_(isinstance(m.kernel_params, dict))
    assert_(isinstance(m.nn_params, dict))


@pytest.mark.parametrize("num_latents", [1, 2])
@pytest.mark.parametrize("num_tasks", [2, 3])
@pytest.mark.parametrize("data_kernel", ['RBF', 'Matern'])
def test_fit_predict_multitask(data_kernel, num_tasks, num_latents):
    rng_key = get_keys()[0]
    X, y = get_dummy_data()
    X = attach_indices(X, num_tasks)
    m = viMTDKL(X.shape[-1] - 1, 2, data_kernel, num_latents=num_latents, shared_input_space=False)
    m.fit(rng_key, X, y, num_steps=10)
    X_test, _ = get_dummy_data()
    X_test = jnp.column_stack([X_test, jnp.ones(len(X_test))])
    mean, var = m.predict(rng_key, X_test)
    assert_(isinstance(mean, jnp.ndarray))
    assert_(isinstance(var, jnp.ndarray))
    assert_equal(len(mean), len(X_test))
    assert_equal(len(var), len(X_test))
