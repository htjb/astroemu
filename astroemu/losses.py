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


@jax.jit
def kl(
    predictions: jnp.ndarray, targets: jnp.ndarray, noise: jnp.ndarray
) -> jnp.ndarray:
    """Kullback-Leibler divergence loss.

    From https://ui.adsabs.harvard.edu/abs/2025MNRAS.544..375B/abstract.

    Args:
        predictions (jnp.ndarray): Predicted probability distributions.
        targets (jnp.ndarray): Target probability distributions.
        noise (jnp.ndarray): Some estimate of noise in the data.

    Returns:
        jnp.ndarray: Scalar KL divergence loss.
    """
    rmse = jnp.sqrt(jnp.mean((predictions - targets) ** 2))
    return predictions.shape[1] / 2 * (rmse / noise) ** 2
