import torch

class standardise:
    def __init__(self, y_mean: torch.Tensor, y_std: torch.Tensor, 
                 x_mean: torch.Tensor, x_std: torch.Tensor):
        """
        Standardises the spectrum and input parameters.
        
        Args:
            lam (array-like): The wavelength values.
            y_mean (float): Mean of the spectrum for standardisation.
            y_std (float): Standard deviation of the spectrum for standardisation.
            x_mean (float): Mean of the input parameters for standardisation.
            x_std (float): Standard deviation of the input parameters for standardisation.
        
        Returns:
            tuple: Standardised spectrum and input parameters.
        """
        self.y_mean = y_mean
        self.y_std = y_std
        self.x_mean = x_mean
        self.x_std = x_std

    def forward(self, y, x=None):
        y = (y - self.y_mean) / self.y_std
        if x is not None:
            x = (x - self.x_mean) / self.x_std
        return y, x

    def backward(self, y, x):
        y = y * self.y_std + self.y_mean
        if x is not None:
            x = x * self.x_std + self.x_mean
        return y, x

class log_base_10():
    def __init__(self, yselector=None, xselector=None, eps=1e-8):
        """
        Logarithm base 10 transformation for numerical stability.

        Args:
            yselector (list, optional): columns of the spectrum to apply log transformation.
                Assumes y is a 2D tensor of [:, N] where N is the number of different spectra
                dependent on x.
            xselector (list, optional): columns of the input parameters to apply log transformation.
        """
        self.yselector = yselector
        self.xselector = xselector
        self.eps = eps

    def forward(self, y: torch.Tensor, x: torch.Tensor = None) -> (
            tuple[torch.Tensor, torch.Tensor]):
        
        if x is not None:
            if self.xselector is not None:
                for i in self.xselector:
                    x[:, i] = torch.log10(x[:, i] + self.eps)
            else:
                x = torch.log10(x + self.eps)
        
        if self.yselector is not None:
            for i in self.yselector:
                y[:, i] = torch.log10(y[:, i] + self.eps)
        else:
            y = torch.log10(y + self.eps)
        
        return y, x
    
    def backward(self, y: torch.Tensor, x: torch.Tensor = None) -> (
            tuple[torch.Tensor, torch.Tensor]):
        
        if x is not None:
            if self.xselector is not None:
                for i in self.xselector:
                    x[:, i] = torch.pow(10, x[:, i]) - self.eps
            else:
                x = torch.pow(10, x) - self.eps
        
        if self.yselector is not None:
            for i in self.yselector:
                y[:, i] = torch.pow(10, y[:, i]) - self.eps
        else:
            y = torch.pow(10, y) - self.eps
        
        return y, x