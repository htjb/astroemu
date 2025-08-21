import torch

class standardise_spectrum:
    def __init__(self, y, input, spec_mean, spec_std, input_mean, input_std):
        """
        Standardises the spectrum and input parameters.
        
        Args:
            y (array-like): The spectrum values.
            input (array-like): The input parameters.
            lam (array-like): The wavelength values.
            spec_mean (float): Mean of the spectrum for standardisation.
            spec_std (float): Standard deviation of the spectrum for standardisation.
            input_mean (float): Mean of the input parameters for standardisation.
            input_std (float): Standard deviation of the input parameters for standardisation.
        
        Returns:
            tuple: Standardised spectrum and input parameters.
        """
        self.spec_mean = spec_mean
        self.spec_std = spec_std
        self.input_mean = input_mean
        self.input_std = input_std

    def forward(self, y, input):
        y = (y - self.spec_mean) / self.spec_std
        input = (input - self.input_mean) / self.input_std
        return y, input

    def backward(self, y, input):
        y = y * self.spec_std + self.spec_mean
        input = input * self.input_std + self.input_mean
        return y, input

def log_base_10(x):
    """
    Computes the logarithm base 10 of x, handling zero values.
    
    Args:
        x (array-like): Input values.
    
    Returns:
        array-like: Logarithm base 10 of x, with a small constant added to avoid log(0).
    """
    def forward(x):
        return torch.log10(torch.tensor(x) + 1e-10)  # Add small constant to avoid log(0)
    
    def backward(x):
        return torch.pow(10, x) - 1e-10