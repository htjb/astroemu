"""Normalisation pipelines for emu package."""

import jax.numpy as jnp


class NormalisationPipeline:
    """Base class for normalisation pipelines."""

    def forward(
        self, y: jnp.ndarray, x: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Apply forward transformation."""
        raise NotImplementedError

    def backward(
        self, y: jnp.ndarray, x: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
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
    ) -> None:
        """Standardises the spectrum and input parameters.

        Args:
            y_mean (float): Mean of the spectrum for standardisation.
            y_std (float): Standard deviation of the spectrum for
                standardisation.
            x_mean (float): Mean of the input parameters for standardisation.
            x_std (float): Standard deviation of the input parameters
                for standardisation.

        Returns:
            tuple: Standardised spectrum and input parameters.
        """
        self.y_mean = y_mean
        self.y_std = y_std
        self.x_mean = x_mean
        self.x_std = x_std

    def forward(
        self, y: jnp.ndarray, x: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Standardise the spectrum and input parameters.

        Args:
            y (jnp.ndarray): Spectrum array.
            x (jnp.ndarray, optional): Input parameters array.
            Defaults to None.

        Returns:
            tuple: Standardised spectrum and input parameters.
        """
        y = (y - self.y_mean) / self.y_std
        if x is not None:
            x = (x - self.x_mean) / self.x_std
        return y, x

    def backward(
        self, y: jnp.ndarray, x: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Destandardise the spectrum and input parameters.

        Args:
            y (jnp.ndarray): Standardised spectrum array.
            x (jnp.ndarray, optional): Standardised input parameters array.
                Defaults to None.

        Returns:
            tuple: Destandardised spectrum and input parameters.
        """
        y = y * self.y_std + self.y_mean
        if x is not None:
            x = x * self.x_std + self.x_mean
        return y, x


class log_base_10(NormalisationPipeline):
    """Logarithm base 10 transformation for numerical stability."""

    def __init__(
        self,
        yselector: list[int] | None = None,
        xselector: list[int] | None = None,
        eps: float = 1e-15,
    ) -> None:
        """Logarithm base 10 transformation for numerical stability.

        Args:
            yselector (list[int] | None): columns of the spectrum to
                apply log transformation.
                Assumes that the spectra are in the last dimension.
            xselector (list[int] | None): columns of the input parameters to
                apply log transformation.
            eps (float): small value to add to avoid log(0).
        """
        self.yselector = yselector
        self.xselector = xselector
        self.eps = eps

    def forward(
        self, y: jnp.ndarray, x: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Apply log10 transformation to selected columns.

        Args:
            y (jnp.ndarray): Spectrum array.
            x (jnp.ndarray, optional): Input parameters array.
                Defaults to None.

        Returns:
            tuple: Transformed spectrum and input parameters.
        """
        if x is not None:
            if self.xselector is not None:
                for i in self.xselector:
                    x = x.at[..., i].set(jnp.log10(x[..., i] + self.eps))
            else:
                x = jnp.log10(x + self.eps)

        if self.yselector is not None:
            for i in self.yselector:
                y = y.at[..., i].set(jnp.log10(y[..., i] + self.eps))
        else:
            y = jnp.log10(y + self.eps)

        return y, x

    def backward(
        self, y: jnp.ndarray, x: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Apply inverse log10 transformation to selected columns.

        Args:
            y (jnp.ndarray): Transformed spectrum array.
            x (jnp.ndarray, optional): Transformed input parameters array.
                Defaults to None.

        Returns:
            tuple: Inverse transformed spectrum and input parameters.
        """
        if x is not None:
            if self.xselector is not None:
                for i in self.xselector:
                    x = x.at[..., i].set(
                        jnp.power(10, x[..., i]) - self.eps
                    )
            else:
                x = jnp.power(10, x) - self.eps

        if self.yselector is not None:
            for i in self.yselector:
                y = y.at[..., i].set(jnp.power(10, y[..., i]) - self.eps)
        else:
            y = jnp.power(10, y) - self.eps

        return y, x
