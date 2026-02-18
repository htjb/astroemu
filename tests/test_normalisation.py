"""Testing the normalisation functions."""

import glob

import jax.numpy as jnp
import pytest

from astroemu.dataloaders import SpectrumDataset, load_spectrum
from astroemu.normalisation import log_base_10, standardise
from astroemu.utils import compute_mean_std


def test_log_10_norm() -> None:
    """Test the SpectrumDataset class."""
    spec = load_spectrum("tests/example_data/sample_000001.npz")
    y = spec["power"]
    x = spec["astro_params"].item()
    x = jnp.array([x[k] for k in x.keys()])

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


def test_standardise() -> None:
    """Test the standardise normalisation pipeline."""
    specloader = SpectrumDataset(
        files=glob.glob("tests/example_data/sample_*.npz"),
        x="k",
        y="power",
        variable_input=["astro_params", "cosmo_params"],
        tiling=False,  # turn off tiling for testing
    )

    y_mean, y_std, x_mean, x_std = compute_mean_std(
        specloader.get_batch_iterator(32)
    )

    normaliser = standardise(
        y_mean=y_mean,
        y_std=y_std,
        x_mean=x_mean,
        x_std=x_std,
        standardise_x=True,
        standardise_y=True,
    )

    all_y, all_x = [], []
    for y, x in specloader.get_batch_iterator(32):
        standardised_y, standardised_x = normaliser.forward(y, x)
        all_y.append(standardised_y)
        all_x.append(standardised_x)

    all_y = jnp.concatenate(all_y, axis=0)
    all_x = jnp.concatenate(all_x, axis=0) 

    assert jnp.allclose(
        jnp.mean(all_y, axis=0), jnp.zeros_like(all_y.shape[-1]), atol=1e-5
    ), "Mean of standardised y should be close to 0."
    assert jnp.allclose(
        jnp.std(all_y, axis=0), jnp.ones_like(all_y.shape[-1]), atol=1e-5
    ), "Std of standardised y should be close to 1."
    assert jnp.allclose(
        jnp.mean(all_x, axis=0), jnp.zeros_like(all_x.shape[-1]), atol=1e-5
    ), "Mean of standardised x should be close to 0."
    assert jnp.allclose(
        jnp.std(all_x, axis=0), jnp.ones_like(all_x.shape[-1]), atol=1e-5
    ), "Std of standardised x should be close to 1."
