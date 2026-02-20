"""Testing the dataloaders."""

import jax.numpy as jnp

from astroemu.dataloaders import SpectrumDataset, load_spectrum
from astroemu.normalisation import log_base_10


def test_load_spectrum() -> None:
    """Test loading a spectrum from .npz file."""
    expected_keys = {
        "k",
        "power",
        "astro_params",
        "cosmo_params",
        "redshfit",  # spelt wrong in the .npz files.
        "code",
        "code_version",
    }

    # Load the spectrum
    loaded_data = load_spectrum(
        "tests/example_data/sample_000001.npz", allow_pickle=True
    )

    assert isinstance(loaded_data, dict), "Loaded data should be a dictionary."
    assert set(loaded_data.keys()) == set(expected_keys), (
        f"Loaded data keys should be {expected_keys}."
    )


def test_spectrum_dataset() -> None:
    """Test the SpectrumDataset class."""
    files = [
        "tests/example_data/sample_000001.npz",
        "tests/example_data/sample_000002.npz",
    ]
    variable_input = ["astro_params", "cosmo_params"]
    logger = log_base_10(log_all_x=True, log_all_y=True, log_all_params=True)
    dataset = SpectrumDataset(
        files=files,
        x="k",
        y="power",
        forward_pipeline=logger,
        variable_input=variable_input,
        tiling=False,
        allow_pickle=True,
    )

    assert len(dataset) == len(files), (
        "Dataset length should match number of files."
    )
    assert dataset.x == "k", "Dataset x key should be 'k'."

    batch_size = 32

    # tiling=False yields (specs, x, inputs)
    y, x, params = next(dataset.get_batch_iterator(batch_size=batch_size))

    assert y.shape[0] <= batch_size, (
        "Batch size should be less than or equal to specified batch_size."
    )
    assert x.shape[0] <= batch_size, (
        "Batch size should be less than or equal to specified batch_size."
    )
    assert params.shape[0] <= batch_size, (
        "Batch size should be less than or equal to specified batch_size."
    )

    # expected shapes after tiling: specs flattened, inputs tiled + x prepended
    tiled_y_shape = jnp.prod(jnp.array(y.shape))
    tiled_x_shape = (jnp.prod(jnp.array(y.shape)), params.shape[-1] + 1)

    dataset.tiling = True  # turn on tiling for testing
    y, x = next(dataset.get_batch_iterator(batch_size=batch_size))

    assert len(y) == tiled_y_shape, (
        "Tiling should flatten the spectrum batches."
    )
    assert x.shape == tiled_x_shape, (
        "Tiled input shape should be (tiled_y_shape, n_params + 1)."
    )


def test_spectrum_dataset_no_variable_input() -> None:
    """Test SpectrumDataset with variable_input=None.

    When variable_input is not provided, __getitem__ should auto-collect
    all non-(x,y) keys, merging dict-valued entries and including only
    numeric scalar entries while skipping string metadata like 'code'.
    """
    files = [
        "tests/example_data/sample_000001.npz",
        "tests/example_data/sample_000002.npz",
    ]
    dataset = SpectrumDataset(
        files=files,
        x="k",
        y="power",
        variable_input=None,
        tiling=False,
        allow_pickle=True,
    )

    # Should not raise even though .npz files contain string metadata
    y, x, params = dataset[0]

    assert params.ndim == 1, "Params should be a 1-D array."
    assert params.dtype == jnp.float32, "Params should be float32."
    # String keys ('code', 'code_version') must be excluded
    assert params.shape[0] > 0, "Params array should not be empty."
