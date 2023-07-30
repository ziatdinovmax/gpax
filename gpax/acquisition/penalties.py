from typing import Optional
import jax
import jax.numpy as jnp


def compute_penalty(X: jnp.ndarray, recent_points: jnp.ndarray,
                    penalty_type: str = "delta", penalty_factor: float = 1.0):
    """
    Compute penalties for points in X for Bayesian Optimization's acquisition function 
    based on a specified penalty type.

    Args:

        X: The array of points in the domain of the objective function for which to compute penalties.
        recent_points: The array of recently evaluated points to consider when computing penalties.
        penalty_type: The type of penalty to apply. Must be either 'delta' or 'inverse_distance'.
        Defaults to 'delta'.
        penalty_factor: The constant multiplier for the 'inverse_distance' penalty. Ignored if penalty_type is 'delta'.
        Defaults to 1.

    Returns:

        The array of computed penalties for each point in X, to be applied to
        the acquisition function values.
    """
    p = penalty_factor
    if penalty_type not in ["delta", "inverse_distance", "inverse distance"]:
        raise NotImplementedError(
            "Avaialble penalty types are 'delta' and 'inverse distance'")
    if penalty_type == "delta":
        penalties = find_and_replace_point_indices(X, recent_points)
    else:
        penalties = p * jax.vmap(penalty_point, in_axes=(0, None))(X, recent_points)
    return penalties


def penalty_point(x: jnp.ndarray, recent_points: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a penalty for point x based on its distance to recent points.
    """
    if recent_points.ndim == 1:
        recent_points = recent_points[:, None] 
    distances = jnp.linalg.norm(recent_points - x, axis=1)
    # Penalties are inversely proportional to distance and timestamp
    if len(recent_points) == 1:
        timestamps = 1
    else:
        timestamps = jnp.arange(len(recent_points) + 1, 1, -1)
    penalties = 1 / (distances + 1) / timestamps
    return jnp.sum(penalties)


def find_and_replace_point_indices(points, other_points):
    # Create an array of zeros
    zero_array = jnp.zeros(len(points))

    for single_point in other_points:
        # Check where in points each element of single_point exists
        index = jnp.where(jnp.all(points == single_point, axis=1))

        # If index array is not empty, replace corresponding zero_array element with jnp.inf
        if index[0].size > 0:
            zero_array = zero_array.at[index[0][0]].set(jnp.inf)

    # Return the modified array
    return zero_array
