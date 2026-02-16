"""Utility functions for emu package."""

from collections.abc import Iterator

import jax.numpy as jnp


def compute_mean_std(
    loader: Iterator[tuple[jnp.ndarray, jnp.ndarray]],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Memory safe mean and std computation.

    Args:
        loader: Iterable yielding (spec, input) where:
            - spec: [batch_size, 5000]
            - input: [batch_size, 5000, N]

    Returns:
        mean_spec: [5000] - mean across batches
        std_spec: [5000] - std across batches
        mean_input: [N] - mean across batches and the 5000 dimension
        std_input: [N] - std across batches and the 5000 dimension
    """
    # Accumulators
    spec_sum = None
    spec_sum_sq = None
    input_sum = None
    input_sum_sq = None
    n_spec_samples = 0
    n_input_samples = 0

    for spec, input_data in loader:
        batch_size = spec.shape[0]

        # === Process spec ===
        # spec shape: [batch_size, 5000] -> we want stats across batch dim
        if spec_sum is None:
            spec_sum = jnp.zeros(spec.shape[1], dtype=spec.dtype)
            spec_sum_sq = jnp.zeros(spec.shape[1], dtype=spec.dtype)

        spec_sum = spec_sum + spec.sum(axis=0)  # sum across batch
        spec_sum_sq = spec_sum_sq + (spec**2).sum(axis=0)  # sum of squares across batch
        n_spec_samples += batch_size

        # === Process input ===
        # input shape: [batch_size, 5000, N] -> we want stats across
        # batch and 5000 dims
        input_flat = input_data.reshape(
            -1, input_data.shape[-1]
        )  # [batch_size * 5000, N]

        if input_sum is None:
            input_sum = jnp.zeros(input_data.shape[-1], dtype=input_data.dtype)
            input_sum_sq = jnp.zeros(input_data.shape[-1], dtype=input_data.dtype)

        input_sum = input_sum + input_flat.sum(
            axis=0
        )  # sum across flattened batch*5000 dim
        input_sum_sq = input_sum_sq + (input_flat**2).sum(axis=0)  # sum of squares
        n_input_samples += input_flat.shape[0]  # batch_size * 5000

    # Compute means and stds
    mean_spec = spec_sum / n_spec_samples
    var_spec = (spec_sum_sq / n_spec_samples) - (mean_spec**2)
    std_spec = jnp.where(var_spec <= 1e-3, 1, jnp.sqrt(var_spec))

    mean_input = input_sum / n_input_samples
    var_input = (input_sum_sq / n_input_samples) - (mean_input**2)
    std_input = jnp.sqrt(
        jnp.clip(var_input, a_min=1e-8)
    )  # clip for numerical stability

    return mean_spec, std_spec, mean_input, std_input
