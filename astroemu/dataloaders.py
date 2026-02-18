"""Data loaders for emu package."""

import jax
import jax.numpy as jnp
from astroemu.normalisation import NormalisationPipeline


def load_spectrum(file: str) -> dict:
    """Load spectrum data from .npz file.

    Args:
        file (str): Path to .npz file.

    Returns:
        dict: Dictionary containing data from .npz file.
    """
    # Note: We use allow_pickle=True to load the .npz files, which may contain
    # jnp.ndarrays.
    data = jnp.load(file, allow_pickle=True)
    input = {k: data[k] for k in data.files}
    return input


class SpectrumDataset:
    """Dataset for loading spectra from .npz files.

    Allows for optional preprocessing via a forward pipeline and
    selection of variable input parameters.
    """

    def __init__(
        self,
        files: list[str],
        x: str,
        y: str,
        forward_pipeline: NormalisationPipeline
        | list[NormalisationPipeline]
        | None = None,
        variable_input: list[str] | str | None = None,
    ) -> None:
        """Initialize SpectrumDataset.

        Args:
            files (list[str]): List of file paths to .npz files.
            x (str): Key for independent variable in .npz files.
            y (str): Key for dependent variable in .npz files.
            forward_pipeline (Any, optional): Preprocessing pipeline.
                Defaults to None.
            variable_input (list[str] | str | None, optional): Keys
                for variable input parameters.
                If None, all parameters except x and y are used.
                Defaults to None.
        """
        self.files = files
        self.varied_input = variable_input
        self.forward_pipeline = (
            forward_pipeline if isinstance(forward_pipeline, list) 
            else [forward_pipeline] if forward_pipeline is not None 
            else []
        )
        self.x = x
        self.y = y

    def __len__(self) -> int:
        """Return number of files in dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get spectrum and input parameters for given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Tuple of (spectrum,
                input parameters).
        """
        input = load_spectrum(self.files[idx])
        x = jnp.array(input[self.x])
        y = jnp.array(input[self.y])
        if self.varied_input:
            input = jnp.array(
                [input[k].item() for k in self.varied_input],
                dtype=jnp.float32,
            )
        else:
            input = jnp.array(
                [
                    input[k].item()
                    for k in sorted(input.keys())
                    if k not in [self.x, self.y]
                ],
                dtype=jnp.float32,
            )
        input = jnp.tile(
            input, (y.shape[0], 1)
        )  # Ensure input shape matches spec
        input = jnp.concatenate(
            [x[:, None], input], axis=1
        )  # Concatenate wavelength with parameters
        for pipeline in self.forward_pipeline:
            y, input = pipeline.forward(y, input)
        
        return y, input

    def get_batch_iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        key: jax.Array | None = None,
    ):
        """Yield batches of (spec, input) as jnp.ndarray.

        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle indices. Defaults to True.
            key (jax.Array | None): JAX PRNG key for shuffling.
                Required when shuffle=True. Defaults to None.

        Yields:
            tuple[jnp.ndarray, jnp.ndarray]: Batches of (spec, input).
        """
        n = len(self)
        indices = jnp.arange(n)
        if shuffle:
            if key is None:
                key = jax.random.PRNGKey(0)
            indices = jax.random.permutation(key, indices)

        for start in range(0, n, batch_size):
            batch_indices = indices[start : start + batch_size]
            specs, inputs = zip(*[self[int(i)] for i in batch_indices])
            yield jnp.stack(specs), jnp.stack(inputs)