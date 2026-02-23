"""Data loaders for emu package."""

from collections.abc import Generator

import jax
import jax.numpy as jnp

from astroemu.normalisation import NormalisationPipeline


def load_spectrum(file: str, allow_pickle: bool = False) -> dict:
    """Load spectrum data from .npz file.

    Args:
        file (str): Path to .npz file.
        allow_pickle (bool): Whether to allow loading pickled objects.

    Returns:
        dict: Dictionary containing data from .npz file.
    """
    data = jnp.load(file, allow_pickle=allow_pickle)
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
        tiling: bool = True,
        allow_pickle: bool = False,
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
            tiling (bool, optional): Whether to tile input/output parameters.
                This is True by default since this is what makes
                astroemu (and globaemu) tick. However, you might want
                to turn it off if you want to use the dataset for
                something other than
                emulation or if you want to calcualte things like
                rolling averages using astroemu.utils.compute_mean_std. Note
                normalisation is applied before tiling.
                Defaults to True.
            allow_pickle (bool): Whether to allow loading pickled objects
                from .npz files. Defaults to False.
        """
        self.files = files
        self.varied_input = variable_input
        if type(variable_input) is str:
            self.varied_input = [variable_input]
        self.forward_pipeline = (
            forward_pipeline
            if isinstance(forward_pipeline, list)
            else [forward_pipeline]
            if forward_pipeline is not None
            else []
        )
        self.x = x
        self.y = y
        self.tiling = tiling
        self.allow_pickle = allow_pickle

    def __len__(self) -> int:
        """Return number of files in dataset."""
        return len(self.files)

    def __getitem__(
        self, idx: int
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Get spectrum and input parameters for given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Tuple of (spectrum,
                input parameters).
        """
        input = load_spectrum(self.files[idx], allow_pickle=self.allow_pickle)
        x = jnp.array(input[self.x])
        y = jnp.array(input[self.y])
        if self.varied_input:
            raw = [(k, input[k].item()) for k in self.varied_input]
        else:
            raw = [
                (k, input[k].item())
                for k in sorted(input.keys())
                if k not in [self.x, self.y]
            ]

        # Build a single params dict: merge dict-valued entries,
        # add numeric scalars by key, skip non-numeric metadata (e.g. strings).
        params: dict = {}
        for k, val in raw:
            if isinstance(val, dict):
                params.update(val)
            elif isinstance(val, int | float):
                params[k] = val

        input = jnp.array(list(params.values()), dtype=jnp.float32)

        return y, x, input

    def get_batch_iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        key: jax.Array | None = None,
    ) -> Generator:
        """Yield batches of spectra and inputs as jnp.ndarray.

        When tiling=True, yields (specs_flat, concat_inputs) where
        specs_flat has shape (batch * len_x,) and concat_inputs has shape
        (batch * len_x, n_params + 1) with x prepended as the first column.

        When tiling=False, yields (specs, x, inputs) with shapes
        (batch, len_x), (batch, len_x), and (batch, n_params) respectively.
        This mode is suitable for computing rolling statistics via
        astroemu.utils.compute_mean_std and building
        normalisation pipelines.

        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle indices. Defaults to True.
                When tiling=True, this also shuffles the tiled samples
                within each batch so the network doesn't see x values
                in sequential order.
            key (jax.Array | None): JAX PRNG key for shuffling.
                Required when shuffle=True. Defaults to None.

        Yields:
            tiling=True:  tuple[jnp.ndarray, jnp.ndarray]
                (specs_flat, concat_inputs)
            tiling=False: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
                (specs, x, inputs)
        """
        n = len(self)
        indices = jnp.arange(n)
        if shuffle:
            if key is None:
                key = jax.random.PRNGKey(0)
            indices = jax.random.permutation(key, indices)

        for start in range(0, n, batch_size):
            batch_indices = indices[start : start + batch_size]
            specs, x, inputs = zip(*[self[int(i)] for i in batch_indices])
            specs = jnp.stack(specs)
            x = jnp.stack(x)
            inputs = jnp.stack(inputs)

            for pipeline in self.forward_pipeline:
                specs, x, inputs = pipeline.forward(specs, x, inputs)

            if self.tiling:
                # tile params to match each x point, then prepend x column
                inputs = jnp.repeat(inputs, repeats=specs.shape[-1], axis=0)
                inputs = jnp.concatenate(
                    [x.flatten()[:, None], inputs], axis=-1
                )
                specs_flat = specs.flatten()

                # Shuffle tiled samples so network doesn't see x
                # values in order
                if shuffle:
                    key, subkey = jax.random.split(key)
                    perm = jax.random.permutation(subkey, len(specs_flat))
                    specs_flat = specs_flat[perm]
                    inputs = inputs[perm]

                yield specs_flat, inputs
            else:
                yield specs, x, inputs
