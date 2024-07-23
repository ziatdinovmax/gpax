import sys
import pytest
import haiku as hk
import jax
import jax.numpy as jnp

sys.path.insert(0, "../gpax/")

from gpax.models.nets import HaikuMLP


def create_model(hidden_dims, output_dim, activation):
    def forward_fn(x):
        model = HaikuMLP(hidden_dims, output_dim, activation)
        return model(x)
    return hk.transform(forward_fn)


@pytest.fixture
def model():
    hidden_dims = [64, 64]
    output_dim = 10
    activation = 'tanh'
    return create_model(hidden_dims, output_dim, activation)


def test_model_initialization(model):
    rng = jax.random.PRNGKey(42)
    input_data = jnp.ones([1, 28*28])
    params = model.init(rng, input_data)
    assert params is not None, "Model parameters should not be None"


def test_model_forward_pass(model):
    rng = jax.random.PRNGKey(42)
    input_data = jnp.ones([1, 28*28])
    params = model.init(rng, input_data)
    output = model.apply(params, rng, input_data)
    assert output.shape == (1, 10), f"Output shape should be (1, 10), but got {output.shape}"
