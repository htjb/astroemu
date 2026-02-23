"""Normalisation pipelines for emu package."""

import warnings

import jax.numpy as jnp


class NormalisationPipeline:
    """Base class for normalisation pipelines."""

    def forward(
        self,
        _y: jnp.ndarray,
        _x: jnp.ndarray,
        _params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply forward transformation."""
        raise NotImplementedError

    def backward(
        self,
        _y: jnp.ndarray,
        _x: jnp.ndarray,
        _params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        params_mean: jnp.ndarray,
        params_std: jnp.ndarray,
        standardise_y: bool = False,
        standardise_x: bool = False,
        standardise_params: bool = False,
    ) -> None:
        """Standardises the spectrum, independent variable, and parameters.

        Args:
            y_mean (jnp.ndarray): Mean of the spectrum.
            y_std (jnp.ndarray): Standard deviation of the spectrum.
            x_mean (float): Mean of the independent variable.
            x_std (float): Standard deviation of the independent variable.
            params_mean (jnp.ndarray): Mean of the input parameters.
            params_std (jnp.ndarray): Standard deviation of the input
                parameters.
            standardise_y (bool): Whether to standardise the spectrum.
                Defaults to False.
            standardise_x (bool): Whether to standardise the independent
                variable. Defaults to False.
            standardise_params (bool): Whether to standardise the input
                parameters. Defaults to False.
        """
        self.y_mean = y_mean
        self.y_std = y_std
        self.x_mean = x_mean
        self.x_std = x_std
        self.params_mean = params_mean
        self.params_std = params_std
        self.standardise_y = standardise_y
        self.standardise_x = standardise_x
        self.standardise_params = standardise_params

    def forward(
        self,
        y: jnp.ndarray,
        x: jnp.ndarray,
        params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Standardise spectrum, independent variable, and input parameters.

        Args:
            y (jnp.ndarray): Spectrum array, shape (batch, len_x).
            x (jnp.ndarray): Independent variable array, shape (batch, len_x).
            params (jnp.ndarray): Input parameters array, shape
                (batch, n_params).

        Returns:
            tuple: Standardised spectrum, independent variable, and parameters.
        """
        if self.standardise_y:
            y = (y - self.y_mean) / self.y_std
        if self.standardise_x:
            x = (x - self.x_mean) / self.x_std
        if self.standardise_params:
            params = (params - self.params_mean) / self.params_std
        return y, x, params

    def backward(
        self,
        y: jnp.ndarray,
        x: jnp.ndarray,
        params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Destandardise spectrum, independent variable, and input parameters.

        Args:
            y (jnp.ndarray): Standardised spectrum array, shape (batch, len_x).
            x (jnp.ndarray): Standardised independent variable, shape
                (batch, len_x).
            params (jnp.ndarray): Standardised input parameters, shape
                (batch, n_params).

        Returns:
            tuple: Destandardised spectrum, independent variable, and
                parameters.
        """
        if self.standardise_y:
            y = y * self.y_std + self.y_mean
        if self.standardise_x:
            x = x * self.x_std + self.x_mean
        if self.standardise_params:
            params = params * self.params_std + self.params_mean
        return y, x, params


class log_base_10(NormalisationPipeline):
    """Logarithm base 10 transformation for numerical stability."""

    def __init__(
        self,
        y_selector: list[int] | jnp.ndarray | None = None,
        x_selector: list[int] | jnp.ndarray | None = None,
        params_selector: list[int] | jnp.ndarray | None = None,
        log_all_y: bool = False,
        log_all_x: bool = False,
        log_all_params: bool = False,
        eps: float = 1e-15,
    ) -> None:
        """Logarithm base 10 transformation for numerical stability.

        Args:
            y_selector (list[int] | None): columns of the spectrum to apply
                log transformation. Assumes spectra are in the last dimension.
                None returns y without any transformation.
            x_selector (list[int] | None): indices of the independent variable
                to apply log transformation. None returns x unchanged.
            params_selector (list[int] | None): columns of the input parameters
                to apply log transformation. Assumes parameters are in the last
                dimension. None returns params without any transformation.
            log_all_y (bool): If True, apply log transformation to all columns
                of the spectrum. Overrides y_selector if True.
            log_all_x (bool): If True, apply log transformation to all elements
                of the independent variable. Overrides x_selector if True.
            log_all_params (bool): If True, apply log transformation to all
                columns of the input parameters. Overrides params_selector.
            eps (float): small value to add to avoid log(0).
        """
        self.y_selector = y_selector
        self.x_selector = x_selector
        self.params_selector = params_selector
        self.log_all_y = log_all_y
        self.log_all_x = log_all_x
        self.log_all_params = log_all_params
        self.eps = eps

        if log_all_y and y_selector is not None:
            warnings.warn("log_all_y is True, overriding y_selector.")
        else:
            if type(self.y_selector) is list:
                self.y_selector = jnp.array(self.y_selector)

        if log_all_x and x_selector is not None:
            warnings.warn("log_all_x is True, overriding x_selector.")
        else:
            if type(self.x_selector) is list:
                self.x_selector = jnp.array(self.x_selector)

        if log_all_params and params_selector is not None:
            warnings.warn(
                "log_all_params is True, overriding params_selector."
            )
        else:
            if type(self.params_selector) is list:
                self.params_selector = jnp.array(self.params_selector)

    def forward(
        self,
        y: jnp.ndarray,
        x: jnp.ndarray,
        params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply log10 transformation to selected columns.

        Args:
            y (jnp.ndarray): Spectrum array, shape (batch, len_x).
            x (jnp.ndarray): Independent variable array, shape (batch, len_x).
            params (jnp.ndarray): Input parameters array, shape
                (batch, n_params).

        Returns:
            tuple: Transformed spectrum, independent variable, and parameters.
        """
        if self.log_all_y:
            y = jnp.log10(y + self.eps)
        elif self.y_selector is not None:
            mask = (
                jnp.zeros(y.shape[-1], dtype=bool)
                .at[self.y_selector]
                .set(True)
            )
            y = jnp.where(mask, jnp.log10(y + self.eps), y)

        if self.log_all_x:
            x = jnp.log10(x + self.eps)
        elif self.x_selector is not None:
            mask = (
                jnp.zeros(x.shape[-1], dtype=bool)
                .at[self.x_selector]
                .set(True)
            )
            x = jnp.where(mask, jnp.log10(x + self.eps), x)

        if self.log_all_params:
            params = jnp.log10(params + self.eps)
        elif self.params_selector is not None:
            mask = (
                jnp.zeros(params.shape[-1], dtype=bool)
                .at[self.params_selector]
                .set(True)
            )
            params = jnp.where(mask, jnp.log10(params + self.eps), params)

        return y, x, params

    def backward(
        self,
        y: jnp.ndarray,
        x: jnp.ndarray,
        params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply inverse log10 transformation to selected columns.

        Args:
            y (jnp.ndarray): Transformed spectrum array, shape (batch, len_x).
            x (jnp.ndarray): Transformed independent variable, shape
                (batch, len_x).
            params (jnp.ndarray): Transformed input parameters, shape
                (batch, n_params).

        Returns:
            tuple: Inverse transformed spectrum, independent variable, and
                parameters.
        """
        if self.log_all_y:
            y = 10**y - self.eps
        elif self.y_selector is not None:
            mask = (
                jnp.zeros(y.shape[-1], dtype=bool)
                .at[self.y_selector]
                .set(True)
            )
            y = jnp.where(mask, 10**y - self.eps, y)

        if self.log_all_x:
            x = 10**x - self.eps
        elif self.x_selector is not None:
            mask = (
                jnp.zeros(x.shape[-1], dtype=bool)
                .at[self.x_selector]
                .set(True)
            )
            x = jnp.where(mask, 10**x - self.eps, x)

        if self.log_all_params:
            params = 10**params - self.eps
        elif self.params_selector is not None:
            mask = (
                jnp.zeros(params.shape[-1], dtype=bool)
                .at[self.params_selector]
                .set(True)
            )
            params = jnp.where(mask, 10**params - self.eps, params)

        return y, x, params
