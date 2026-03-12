"""Tests for loss functions in astroemu.losses."""

import jax.numpy as jnp
import pytest

from astroemu.losses import kl, mse

# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------


def test_mse_perfect_predictions() -> None:
    """MSE is zero when predictions exactly match targets."""
    preds = jnp.array([1.0, 2.0, 3.0])
    targets = jnp.array([1.0, 2.0, 3.0])
    assert float(mse(preds, targets)) == 0.0


def test_mse_known_value() -> None:
    """MSE equals the analytically computed value for known inputs."""
    # mean((0-1)^2, (0-1)^2, (0-1)^2, (0-1)^2) == 1.0
    assert float(mse(jnp.zeros(4), jnp.ones(4))) == pytest.approx(1.0)


def test_mse_known_value_mixed() -> None:
    """MSE is correct for mixed positive and negative residuals."""
    # residuals: [-1, 1, -1, 1] -> mean(1,1,1,1) = 1.0
    preds = jnp.array([0.0, 2.0, 0.0, 2.0])
    targets = jnp.array([1.0, 1.0, 1.0, 1.0])
    assert float(mse(preds, targets)) == pytest.approx(1.0)


def test_mse_returns_scalar() -> None:
    """MSE output is a scalar (shape ())."""
    result = mse(jnp.ones(10), jnp.zeros(10))
    assert result.shape == ()


def test_mse_symmetric() -> None:
    """MSE(preds, targets) == MSE(targets, preds)."""
    preds = jnp.array([1.0, 3.0, 5.0])
    targets = jnp.array([2.0, 2.0, 2.0])
    assert float(mse(preds, targets)) == pytest.approx(
        float(mse(targets, preds))
    )


# ---------------------------------------------------------------------------
# KL
# ---------------------------------------------------------------------------


def test_kl_perfect_predictions() -> None:
    """KL loss is zero when predictions exactly match targets."""
    preds = jnp.ones((3, 5))
    targets = jnp.ones((3, 5))
    assert float(kl(preds, targets, noise=1.0)) == pytest.approx(0.0)


def test_kl_known_value() -> None:
    """KL equals the analytically computed value for known inputs.

    predictions = ones((2, 4)), targets = zeros((2, 4)), noise = 1.0
    rmse = sqrt(mean(1^2 * 8)) = 1.0
    kl   = ndata/2 * (rmse/noise)^2 = 4/2 * 1^2 = 2.0
    """
    preds = jnp.ones((2, 4))
    targets = jnp.zeros((2, 4))
    assert float(
        kl(preds, targets, noise=1.0, ndata=preds.shape[1])
    ) == pytest.approx(2.0)


def test_kl_returns_scalar() -> None:
    """KL output is a scalar (shape ())."""
    result = kl(jnp.ones((4, 6)), jnp.zeros((4, 6)), noise=1.0)
    assert result.shape == ()


def test_kl_increases_with_larger_error() -> None:
    """KL loss is larger when the prediction error is larger."""
    targets = jnp.zeros((3, 8))
    small_error = kl(jnp.ones((3, 8)) * 0.1, targets, noise=1.0)
    large_error = kl(jnp.ones((3, 8)) * 1.0, targets, noise=1.0)
    assert float(large_error) > float(small_error)


def test_kl_decreases_with_larger_noise() -> None:
    """KL loss is smaller when noise is larger (same prediction error)."""
    preds = jnp.ones((3, 8))
    targets = jnp.zeros((3, 8))
    loss_low_noise = kl(preds, targets, noise=1.0)
    loss_high_noise = kl(preds, targets, noise=10.0)
    assert float(loss_high_noise) < float(loss_low_noise)


def test_kl_scales_with_output_dimension() -> None:
    """KL loss scales linearly with ndata."""
    targets = jnp.zeros((2, 4))
    preds_4 = jnp.ones((2, 4))
    preds_8 = jnp.ones((2, 8))
    targets_8 = jnp.zeros((2, 8))
    loss_4 = float(kl(preds_4, targets, noise=1.0, ndata=preds_4.shape[1]))
    loss_8 = float(kl(preds_8, targets_8, noise=1.0, ndata=preds_8.shape[1]))
    assert loss_8 == pytest.approx(loss_4 * 2.0)
