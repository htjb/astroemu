import jax.numpy as jnp
import torch
from torch.utils.data import Dataset

def load_spectrum(file):
    data = jnp.load(file)
    input = {k: data[k] for k in data.files}
    return input

class SpectrumDataset(Dataset):
    def __init__(self, files, x, y, forward_pipeline=None, variable_input=None):
        self.files = files
        self.varied_input = variable_input
        self.forward_pipeline = forward_pipeline
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input = load_spectrum(self.files[idx])
        x = torch.tensor(input[self.x])
        y = torch.tensor(input[self.y])
        if self.varied_input:
            input = torch.tensor([input[k].item() for k in self.varied_input], dtype=torch.float32)
        else:
            input = torch.tensor([input[k].item() for k in sorted(input.keys())
                                  if k not in [self.x, self.y]], dtype=torch.float32)
        input = torch.tile(input, (y.shape[0], 1))  # Ensure input shape matches spec
        input = torch.cat([x[:, None], input], axis=1)  # Concatenate wavelength with parameters
        if self.forward_pipeline:
            return self.forward_pipeline.forward(y, input)
        else:
            return y, input
    
class NormalizeSpectrumDataset(SpectrumDataset):
    def __init__(self, files, x, y,
                 forward_pipeline=None,
                 super_forward_pipeline=None,
                 variable_input=None):
        super().__init__(files, x, y, 
                         forward_pipeline=super_forward_pipeline, 
                         variable_input=variable_input)
        self.normalize_pipeline = forward_pipeline


    def __getitem__(self, idx):
        y, input = super().__getitem__(idx)
        if self.normalize_pipeline:
            return self.normalize_pipeline.forward(y, input)
        else:
            return y, input