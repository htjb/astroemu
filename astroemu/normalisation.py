"""Normalisation pipelines for emu package."""

import warnings

import jax.numpy as jnp


class NormalisationPipeline:
    """Base class for normalisation pipelines."""

    def forward(
        self, y: jnp.ndarray, x: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply forward transformation."""
        raise NotImplementedError

    def backward(
        self, y: jnp.ndarray, x: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply backward transformation."""
        raise NotImplementedError


class standardise(NormalisationPipeline):
    """Standardisation normalisation pipeline."""

    def __init__(
        self,
        y_mean: jnp.ndarray,
        y_std: jnp.ndarray,
        x_mean: jnp.ndarray,
        x_std: jnp.ndarray,
        standardise_x: bool = False,
        standardise_y: bool = False,
    ) -> None:
        """Standardises the spectrum and input parameters.

        Args:
            y_mean (float): Mean of the spectrum for standardisation.
            y_std (float): Standard deviation of the spectrum for
                standardisation.
            x_mean (float): Mean of the input parameters for standardisation.
            x_std (float): Standard deviation of the input parameters
                for standardisation.
            standardise_x (bool): Whether to standardise the input
                parameters. Defaults to False.
            standardise_y (bool): Whether to standardise the spectrum.
                Defaults to False.

        Returns:
            tuple: Standardised spectrum and input parameters.
        """
        self.y_mean = y_mean
        self.y_std = y_std
        self.x_mean = x_mean
        self.x_std = x_std
        self.standardise_x = standardise_x
        self.standardise_y = standardise_y

    def forward(
        self, y: jnp.ndarray, x: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Standardise the spectrum and input parameters.

        Args:
            y (jnp.ndarray): Spectrum array.
            x (jnp.ndarray): Input parameters array.

        Returns:
            tuple: Standardised spectrum and input parameters.
        """
        if self.standardise_y:
            y = (y - self.y_mean) / self.y_std

        if self.standardise_x:
            x = (x - self.x_mean) / self.x_std
        return y, x

    def backward(
        self, y: jnp.ndarray, x: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Destandardise the spectrum and input parameters.

        Args:
            y (jnp.ndarray): Standardised spectrum array.
            x (jnp.ndarray): Standardised input parameters array.

        Returns:
            tuple: Destandardised spectrum and input parameters.
        """
        if self.standardise_y:
            y = y * self.y_std + self.y_mean

        if self.standardise_x:
            x = x * self.x_std + self.x_mean
        return y, x


class log_base_10(NormalisationPipeline):
    """Logarithm base 10 transformation for numerical stability."""

    def __init__(
        self,
        yselector: list[int] | None = None,
        xselector: list[int] | None = None,
        log_all_y: bool = False,
        log_all_x: bool = False,
        eps: float = 1e-15,
    ) -> None:
        """Logarithm base 10 transformation for numerical stability.

        Args:
            yselector (list[int] | None): columns of the spectrum to
                apply log transformation.
                Assumes that the spectra are in the last dimension.
                None returns y without any transformation.
            xselector (list[int] | None): columns of the input parameters to
                apply log transformation.
                Assumes that the input parameters are in the last dimension.
                None returns x without any transformation.
            log_all_y (bool): If True, apply log transformation to all
                columns of the spectrum. Overrides yselector if True.
            log_all_x (bool): If True, apply log transformation to all
                columns of the input parameters. Overrides xselector if True.
            eps (float): small value to add to avoid log(0).
        """
        self.yselector = yselector
        self.xselector = xselector
        self.log_all_y = log_all_y
        self.log_all_x = log_all_x
        self.eps = eps

        if log_all_y and yselector is not None:
            warnings.warn("log_all_y is True, overriding yselector.")
        
        if log_all_x and xselector is not None:
            warnings.warn("log_all_x is True, overriding xselector.")

    @staticmethod
    def _apply_log10(
        arr: jnp.ndarray, selector: list[int] | None, eps: float
    ) -> jnp.ndarray:
        if selector is None:
            return arr  # skip
        mask = jnp.zeros(arr.shape[-1], dtype=bool).at[selector].set(True)
        return jnp.where(mask, jnp.log10(arr + eps), arr)

    def forward(
        self, y: jnp.ndarray, x: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply log10 transformation to selected columns.

        Args:
            y (jnp.ndarray): Spectrum array.
            x (jnp.ndarray): Input parameters array.
                Defaults to None.

        Returns:
            tuple: Transformed spectrum and input parameters.
        """
        if self.log_all_x:
            x = jnp.log10(x + self.eps)
        elif self.xselector is not None:
            mask = (
                jnp.zeros(x.shape[-1], dtype=bool).at[self.xselector].set(True)
            )
            x = jnp.where(mask, jnp.log10(x + self.eps), x)

        if self.log_all_y:
            y = jnp.log10(y + self.eps)
        elif self.yselector is not None:
            mask = (
                jnp.zeros(y.shape[-1], dtype=bool).at[self.yselector].set(True)
            )
            y = jnp.where(mask, jnp.log10(y + self.eps), y)

        return y, x

    def backward(
        self, y: jnp.ndarray, x: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply inverse log10 transformation to selected columns.

        Args:
            y (jnp.ndarray): Transformed spectrum array.
            x (jnp.ndarray): Transformed input parameters array.
                Defaults to None.

        Returns:
            tuple: Inverse transformed spectrum and input parameters.
        """
        if self.log_all_x:
            x = 10**x - self.eps
        elif self.xselector is not None:
            mask = (
                jnp.zeros(x.shape[-1], dtype=bool).at[self.xselector].set(True)
            )
            x = jnp.where(mask, 10**x - self.eps, x)

        if self.log_all_y:
            y = 10**y - self.eps
        elif self.yselector is not None:
            mask = (
                jnp.zeros(y.shape[-1], dtype=bool).at[self.yselector].set(True)
            )
            y = jnp.where(mask, 10**y - self.eps, y)

        return y, x
