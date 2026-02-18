"""Testing the normalisation functions."""

import pytest

import jax.numpy as jnp

from astroemu.utils import compute_mean_std
from astroemu.normalisation import log_base_10, standardise
from astroemu.dataloaders import load_spectrum


def test_log_10_norm() -> None:
    """Test the SpectrumDataset class."""
    spec = load_spectrum("tests/example_data/sample_000001.npz")
    y = spec["power"]
    x = spec["astro_params"].item()
    x = jnp.array([x[k] for k in x.keys()])
    print(x)

    logger = log_base_10(xselector=jnp.arange(len(x)))
    unchanged_y, logged_params = logger.forward(y, x)

    assert unchanged_y.shape == y.shape, (
        "y after log_base_10 should have same shape as y."
    )
    assert logged_params.shape == x.shape, (
        "x after log_base_10 should have same shape as x."
    )
    assert jnp.allclose(unchanged_y, y), (
        "y should be unchanged by log_base_10 forward transformation"
        + " when yselector is not provided and log_all_y is False."
    )
    assert jnp.allclose(logged_params, jnp.log10(x)), (
        "x after log_base_10 with xselectors should be log10 of x."
    )

    logger = log_base_10()
    unchanged_y, unchanged_x = logger.forward(y, x)

    assert jnp.allclose(unchanged_y, y), (
        "y should be unchanged by log_base_10 forward transformation"
        + "when no yselectors."
    )
    assert jnp.allclose(unchanged_x, x), (
        "x should be unchanged by log_base_10 forward" + ""
        "transformation when no selectors are provided."
    )

    logger = log_base_10(log_all_y=True, log_all_x=True)
    logged_y, logged_x = logger.forward(y, x)

    assert jnp.allclose(logged_y, jnp.log10(y)), (
        "y after log_base_10 with log_all_y should be log10 of y."
    )
    assert jnp.allclose(logged_x, jnp.log10(x)), (
        "x after log_base_10 with log_all_x should be log10 of x."
    )

    with pytest.warns(
        UserWarning, match="log_all_y is True, overriding yselector."
    ):
        logger = log_base_10(yselector=[0, 2], log_all_y=True)
        logged_y, unchanged_x = logger.forward(y, x)

        assert jnp.allclose(logged_y, jnp.log10(y)), (
            "y after log_base_10 with log_all_y should be log10 of y,"
            + "even if yselector is also provided."
        )

    with pytest.warns(
        UserWarning, match="log_all_x is True, overriding xselector."
    ):
        logger = log_base_10(xselector=[0, 2], log_all_x=True)
        unchanged_y, logged_x = logger.forward(y, x)

        assert jnp.allclose(logged_x, jnp.log10(x)), (
            "x after log_base_10 with log_all_x should be log10 of x,"
            + "even if xselector is also provided."
        )
