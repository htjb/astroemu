"""Loss functions for astroemu."""

import jax
import jax.numpy as jnp


@jax.jit
def mse(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss.

    Args:
        predictions (jnp.ndarray): Predicted values.
        targets (jnp.ndarray): Target values.

    Returns:
        jnp.ndarray: Scalar MSE loss.
    """
    return jnp.mean((predictions - targets) ** 2)
