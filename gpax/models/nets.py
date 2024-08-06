from typing import Dict, List, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
import haiku as hk

import optax
from tqdm import tqdm


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
    

class DeterministicNN:
    def __init__(self, model: hk.Transformed, input_dim: int, learning_rate: float = 0.01):
        """
        Initialize the deterministic neural network with an existing Haiku model.

        Args:
        model (hk.Transformed): A Haiku Transformed model ready for training and prediction.
        input_dim (int, optional): Dimension of the input data. Required for model initialization
        learning_rate (float): Learning rate for the optimizer.
        """
        self.learning_rate = learning_rate
        self.model = model
        # Initialize model parameters, requiring a dummy input if input_dim is provided
        self.params = model.init(jax.random.PRNGKey(0), jnp.ones((1, input_dim)))
        self.optimizer = optax.adam(learning_rate=self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def mse_loss(self, params: hk.Params, inputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        predictions = self.model.apply(params, None, inputs)
        return jnp.mean((predictions - targets) ** 2)

    def train_step(self, inputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        loss_value, grads = jax.value_and_grad(self.mse_loss)(self.params, inputs, targets)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss_value

    def train(self, X_train: jnp.ndarray, y_train: jnp.ndarray, epochs: int) -> None:
        with tqdm(total=epochs, desc="Training Progress", leave=True) as pbar:
            for epoch in range(epochs):
                loss = self.train_step(X_train, y_train)
                pbar.update(1)
                pbar.set_postfix_str(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.model.apply(self.params, None, X)

