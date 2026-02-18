"""Testing the normalisation functions."""

import glob

import jax.numpy as jnp
import pytest

from astroemu.dataloaders import SpectrumDataset, load_spectrum
from astroemu.normalisation import log_base_10, standardise
from astroemu.utils import compute_mean_std


def test_log_10_norm() -> None:
    """Test the log_base_10 normalisation pipeline."""
    spec = load_spectrum("tests/example_data/sample_000001.npz")
    y = spec["power"]
    x_freq = spec["k"]
    params_raw = spec["astro_params"].item()
    params = jnp.array([params_raw[k] for k in params_raw.keys()])

    # xselector now refers to the independent variable (x); params_selector
    # refers to input parameters.  Test that params_selector logs params only.
    logger = log_base_10(params_selector=jnp.arange(len(params)))
    unchanged_y, unchanged_x, logged_params = logger.forward(y, x_freq, params)

    assert unchanged_y.shape == y.shape, (
        "y after log_base_10 should have same shape as y."
    )
    assert unchanged_x.shape == x_freq.shape, (
        "x after log_base_10 should have same shape as x."
    )
    assert logged_params.shape == params.shape, (
        "params after log_base_10 should have same shape as params."
    )
    assert jnp.allclose(unchanged_y, y), (
        "y should be unchanged by log_base_10 forward transformation"
        + " when yselector is not provided and log_all_y is False."
    )
    assert jnp.allclose(unchanged_x, x_freq), (
        "x should be unchanged when xselector is not provided."
    )
    assert jnp.allclose(logged_params, jnp.log10(params)), (
        "params after log_base_10 with params_selector should be log10."
    )

    # No selectors — everything passes through unchanged
    logger = log_base_10()
    unchanged_y, unchanged_x, unchanged_params = logger.forward(
        y, x_freq, params
    )

    assert jnp.allclose(unchanged_y, y), (
        "y should be unchanged by log_base_10 when no selectors provided."
    )
    assert jnp.allclose(unchanged_x, x_freq), (
        "x should be unchanged by log_base_10 when no selectors provided."
    )
    assert jnp.allclose(unchanged_params, params), (
        "params should be unchanged by log_base_10 when no selectors provided."
    )

    # log_all flags log everything
    logger = log_base_10(log_all_y=True, log_all_x=True, log_all_params=True)
    logged_y, logged_x, logged_params = logger.forward(y, x_freq, params)

    assert jnp.allclose(logged_y, jnp.log10(y)), (
        "y after log_base_10 with log_all_y should be log10 of y."
    )
    assert jnp.allclose(logged_x, jnp.log10(x_freq)), (
        "x after log_base_10 with log_all_x should be log10 of x."
    )
    assert jnp.allclose(logged_params, jnp.log10(params)), (
        "params after log_base_10 with log_all_params should be log10."
    )

    with pytest.warns(
        UserWarning, match="log_all_y is True, overriding yselector."
    ):
        logger = log_base_10(yselector=[0, 2], log_all_y=True)
        logged_y, unchanged_x, unchanged_params = logger.forward(
            y, x_freq, params
        )

        assert jnp.allclose(logged_y, jnp.log10(y)), (
            "y after log_base_10 with log_all_y should be log10 of y,"
            + "even if yselector is also provided."
        )

    with pytest.warns(
        UserWarning, match="log_all_x is True, overriding xselector."
    ):
        logger = log_base_10(xselector=[0, 2], log_all_x=True)
        unchanged_y, logged_x, unchanged_params = logger.forward(
            y, x_freq, params
        )

        assert jnp.allclose(logged_x, jnp.log10(x_freq)), (
            "x after log_base_10 with log_all_x should be log10 of x,"
            + "even if xselector is also provided."
        )

    with pytest.warns(
        UserWarning,
        match="log_all_params is True, overriding params_selector.",
    ):
        logger = log_base_10(params_selector=[0, 2], log_all_params=True)
        unchanged_y, unchanged_x, logged_params = logger.forward(
            y, x_freq, params
        )

        assert jnp.allclose(logged_params, jnp.log10(params)), (
            "params after log_base_10 with log_all_params should be log10,"
            + "even if params_selector is also provided."
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

    y_mean, y_std, x_mean, x_std, params_mean, params_std = compute_mean_std(
        specloader.get_batch_iterator(32)
    )

    normaliser = standardise(
        y_mean=y_mean,
        y_std=y_std,
        x_mean=x_mean,
        x_std=x_std,
        params_mean=params_mean,
        params_std=params_std,
        standardise_x=True,
        standardise_y=True,
        standardise_params=True,
    )

    all_y, all_x, all_params = [], [], []
    for y, x, params in specloader.get_batch_iterator(32):
        std_y, std_x, std_params = normaliser.forward(y, x, params)
        all_y.append(std_y)
        all_x.append(std_x)
        all_params.append(std_params)

    all_y = jnp.concatenate(all_y, axis=0)
    all_x = jnp.concatenate(all_x, axis=0)
    all_params = jnp.concatenate(all_params, axis=0)

    assert jnp.allclose(
        jnp.mean(all_y, axis=0),
        jnp.zeros_like(all_y.shape[-1]),
        atol=1e-5,
    ), "Mean of standardised y should be close to 0."
    assert jnp.allclose(
        jnp.std(all_y, axis=0),
        jnp.ones_like(all_y.shape[-1]),
        atol=1e-5,
    ), "Std of standardised y should be close to 1."
    assert jnp.allclose(
        jnp.mean(all_params, axis=0),
        jnp.zeros_like(all_params.shape[-1]),
        atol=1e-5,
    ), "Mean of standardised params should be close to 0."
    assert jnp.allclose(
        jnp.std(all_params, axis=0),
        jnp.ones_like(all_params.shape[-1]),
        atol=1e-5,
    ), "Std of standardised params should be close to 1."
