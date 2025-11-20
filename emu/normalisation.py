"""Normalisation pipelines for emu package."""

import torch


class NormalisationPipeline:
    """Base class for normalisation pipelines."""

    def forward(
        self, y: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply forward transformation."""
        raise NotImplementedError

    def backward(
        self, y: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply backward transformation."""
        raise NotImplementedError


class standardise(NormalisationPipeline):
    """Standardisation normalisation pipeline."""

    def __init__(
        self,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
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
        self, y: torch.Tensor, x: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Standardise the spectrum and input parameters.

        Args:
            y (torch.Tensor): Spectrum tensor.
            x (torch.Tensor, optional): Input parameters tensor.
            Defaults to None.

        Returns:
            tuple: Standardised spectrum and input parameters.
        """
        y = (y - self.y_mean) / self.y_std
        if x is not None:
            x = (x - self.x_mean) / self.x_std
        return y, x

    def backward(
        self, y: torch.Tensor, x: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Destandardise the spectrum and input parameters.

        Args:
            y (torch.Tensor): Standardised spectrum tensor.
            x (torch.Tensor, optional): Standardised input parameters tensor.
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
        eps: float | None = 1e-15,
    ) -> None:
        """Logarithm base 10 transformation for numerical stability.

        Args:
            yselector (list[int] | None): columns of the spectrum to
                apply log transformation.
                Assumes that the spectra are in the last dimension.
            xselector (list[int] | None): columns of the input parameters to
                apply log transformation.
            eps (float | None): small value to add to avoid log(0).
        """
        self.yselector = yselector
        self.xselector = xselector
        self.eps = eps

    def forward(
        self, y: torch.Tensor, x: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply log10 transformation to selected columns.

        Args:
            y (torch.Tensor): Spectrum tensor.
            x (torch.Tensor, optional): Input parameters tensor.
                Defaults to None.

        Returns:
            tuple: Transformed spectrum and input parameters.
        """
        if x is not None:
            if self.xselector is not None:
                for i in self.xselector:
                    x[..., i] = torch.log10(x[..., i] + self.eps)
            else:
                x = torch.log10(x + self.eps)

        if self.yselector is not None:
            for i in self.yselector:
                y[..., i] = torch.log10(y[..., i] + self.eps)
        else:
            y = torch.log10(y + self.eps)

        return y, x

    def backward(
        self, y: torch.Tensor, x: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply inverse log10 transformation to selected columns.

        Args:
            y (torch.Tensor): Transformed spectrum tensor.
            x (torch.Tensor, optional): Transformed input parameters tensor.
                Defaults to None.

        Returns:
            tuple: Inverse transformed spectrum and input parameters.
        """
        if x is not None:
            if self.xselector is not None:
                for i in self.xselector:
                    x[..., i] = torch.pow(10, x[..., i]) - self.eps
            else:
                x = torch.pow(10, x) - self.eps

        if self.yselector is not None:
            for i in self.yselector:
                y[..., i] = torch.pow(10, y[..., i]) - self.eps
        else:
            y = torch.pow(10, y) - self.eps

        return y, x
