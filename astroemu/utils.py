"""Utility functions for emu package."""

from collections.abc import Iterator

import jax.numpy as jnp


def compute_mean_std(
    loader: Iterator[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Memory safe mean and std computation.

    Expects the loader to yield three-tuples (spec, x, inputs) as produced
    by SpectrumDataset.get_batch_iterator() with tiling=False.

    Since x (the independent variable) is identical for every sample in the
    dataset, its mean and std are computed as global scalars over all elements
    rather than per-column statistics.

    Args:
        loader: Iterable yielding (spec, x, inputs) where:
            - spec:   (batch_size, len_x)
            - x:      (batch_size, len_x)
            - inputs: (batch_size, n_params)

    Returns:
        mean_spec:   (len_x,)  - per-frequency mean across batches
        std_spec:    (len_x,)  - per-frequency std across batches
        mean_x:      scalar Array - global mean of the independent variable
        std_x:       scalar Array - global std of the independent variable
        mean_input:  (n_params,) - per-parameter mean across batches
        std_input:   (n_params,) - per-parameter std across batches
    """
    spec_sum = None
    spec_sum_sq = None
    x_sum = jnp.zeros(())
    x_sum_sq = jnp.zeros(())
    input_sum = None
    input_sum_sq = None
    n_spec_samples = 0
    n_x_elements = 0
    n_input_samples = 0

    for spec, x, input_data in loader:
        batch_size = spec.shape[0]

        # spectrum accumulators: sum across batch dimension
        if spec_sum is None:
            spec_sum = jnp.zeros(spec.shape[1], dtype=spec.dtype)
            spec_sum_sq = jnp.zeros(spec.shape[1], dtype=spec.dtype)
        spec_sum = spec_sum + spec.sum(axis=0)
        spec_sum_sq = spec_sum_sq + (spec**2).sum(axis=0)
        n_spec_samples += batch_size

        # x accumulator: global scalar (all rows identical, treat as flat)
        x_sum = x_sum + x.sum()
        x_sum_sq = x_sum_sq + (x**2).sum()
        n_x_elements += x.size

        # input parameter accumulators: sum across batch dimension
        if input_sum is None:
            input_sum = jnp.zeros(
                input_data.shape[-1], dtype=input_data.dtype
            )
            input_sum_sq = jnp.zeros(
                input_data.shape[-1], dtype=input_data.dtype
            )
        input_sum = input_sum + input_data.sum(axis=0)
        input_sum_sq = input_sum_sq + (input_data**2).sum(axis=0)
        n_input_samples += batch_size

    mean_spec = spec_sum / n_spec_samples
    var_spec = (spec_sum_sq / n_spec_samples) - mean_spec**2
    std_spec = jnp.where(jnp.sqrt(var_spec) < 1e-3, 1.0, jnp.sqrt(var_spec))

    mean_x = x_sum / n_x_elements
    var_x = (x_sum_sq / n_x_elements) - mean_x**2
    std_x = jnp.where(jnp.sqrt(var_x) < 1e-3, 1.0, jnp.sqrt(var_x))

    mean_input = input_sum / n_input_samples
    var_input = (input_sum_sq / n_input_samples) - mean_input**2
    std_input = jnp.where(
        jnp.sqrt(var_input) < 1e-3, 1.0, jnp.sqrt(var_input)
    )

    return mean_spec, std_spec, mean_x, std_x, mean_input, std_input
