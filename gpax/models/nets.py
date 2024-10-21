from typing import Sequence
import jax
import jax.numpy as jnp
import haiku as hk


class HaikuMLP(hk.Module):
    def __init__(self, hidden_dims: Sequence[int], output_dim: int, activation: str = 'tanh'):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation_fn = jax.nn.tanh if self.activation == 'tanh' else jax.nn.relu

        for i, hidden_dim in enumerate(self.hidden_dims):
            x = hk.Linear(hidden_dim, name=f"Dense{i}")(x)
            x = activation_fn(x)

        if self.output_dim:
            x = hk.Linear(self.output_dim, name=f"Dense{len(self.hidden_dims)}")(x)

        return x
