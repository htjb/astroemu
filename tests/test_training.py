"""Testing the training loop and loss functions."""

import glob

import jax.numpy as jnp

from astroemu.dataloaders import SpectrumDataset
from astroemu.losses import mse
from astroemu.train import train

_FILES = sorted(glob.glob("tests/example_data/sample_*.npz"))
_TRAIN_FILES = _FILES[:80]
_VAL_FILES = _FILES[80:]

_DATASET_KWARGS: dict = dict(
    x="k",
    y="power",
    variable_input=["astro_params", "cosmo_params"],
    tiling=True,
    allow_pickle=True,
)


def _make_datasets() -> tuple[SpectrumDataset, SpectrumDataset]:
    return (
        SpectrumDataset(files=_TRAIN_FILES, **_DATASET_KWARGS),
        SpectrumDataset(files=_VAL_FILES, **_DATASET_KWARGS),
    )


def test_mse() -> None:
    """MSE is zero for perfect predictions and correct for known values."""
    preds = jnp.array([1.0, 2.0, 3.0])
    targets = jnp.array([1.0, 2.0, 3.0])
    assert float(mse(preds, targets)) == 0.0, (
        "MSE should be zero when predictions equal targets."
    )

    # mean((0 - 1)^2, (0 - 1)^2, (0 - 1)^2, (0 - 1)^2) == 1.0
    assert float(mse(jnp.zeros(4), jnp.ones(4))) == 1.0, (
        "MSE should equal 1.0 when predicting zeros for unit targets."
    )


def test_train_output_types() -> None:
    """train() returns (dict, list[float], list[float]) of expected lengths."""
    train_ds, val_ds = _make_datasets()
    n_epochs = 3

    params, train_losses, val_losses = train(
        train_ds,
        val_ds,
        hidden_size=8,
        nlayers=1,
        epochs=n_epochs,
        patience=50,
        batch_size=16,
        key=0,
    )

    assert isinstance(params, dict), "Returned params should be a dict."
    assert isinstance(train_losses, list), "train_losses should be a list."
    assert isinstance(val_losses, list), "val_losses should be a list."
    assert len(train_losses) == n_epochs, (
        "train_losses should have one entry per epoch."
    )
    assert len(val_losses) == n_epochs, (
        "val_losses should have one entry per epoch."
    )
    assert all(isinstance(loss, float) for loss in train_losses), (
        "All train losses should be Python floats."
    )
    assert all(isinstance(loss, float) for loss in val_losses), (
        "All val losses should be Python floats."
    )


def test_train_params_structure() -> None:
    """Returned params dict has the expected keys for the architecture."""
    train_ds, val_ds = _make_datasets()
    nlayers = 2

    params, _, _ = train(
        train_ds,
        val_ds,
        hidden_size=16,
        nlayers=nlayers,
        epochs=2,
        patience=50,
        batch_size=32,
        key=0,
    )

    # initialise_mlp creates (nlayers + 2) layers: 1 input + nlayers hidden
    # + 1 output, each with a weights{i} and bias{i} key.
    n_layers = nlayers + 2
    expected_keys = {f"weights{i}" for i in range(n_layers)} | {
        f"bias{i}" for i in range(n_layers)
    }
    assert set(params.keys()) == expected_keys, (
        f"Params keys should be {expected_keys}, got {set(params.keys())}."
    )


def test_train_early_stopping() -> None:
    """Training terminates at or before max epochs and loss list are synced."""
    train_ds, val_ds = _make_datasets()
    epochs = 50

    _, train_losses, val_losses = train(
        train_ds,
        val_ds,
        hidden_size=8,
        nlayers=1,
        epochs=epochs,
        patience=2,
        batch_size=16,
        key=0,
    )

    assert len(train_losses) == len(val_losses), (
        "train_losses and val_losses should have the same length."
    )
    assert len(train_losses) <= epochs, (
        "Number of epochs run should not exceed the max epochs setting."
    )
    assert len(train_losses) >= 1, "At least one epoch should have run."


def test_train_custom_activation() -> None:
    """Training completes successfully with a non-default activation (tanh)."""
    train_ds, val_ds = _make_datasets()

    params, train_losses, val_losses = train(
        train_ds,
        val_ds,
        hidden_size=8,
        nlayers=1,
        act="tanh",
        epochs=3,
        patience=50,
        batch_size=16,
        key=0,
    )

    assert isinstance(params, dict), (
        "Params should be a dict when using tanh activation."
    )
    assert len(train_losses) == 3, (
        "Should run for exactly 3 epochs with tanh activation."
    )
    assert all(jnp.isfinite(loss) for loss in train_losses), (
        "All losses should be finite with tanh activation."
    )
