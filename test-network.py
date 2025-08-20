from emu.network import initialise_mlp, mlp
from emu.utils import compute_mean_std
import jax
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import optax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import glob

key = random.PRNGKey(0)

base_dir = 'fsps-data1'
files = glob.glob(base_dir + '/*.npz')[:1000]  # Limit to 1000 files for testing

def load_spectrum(file):
    data = jnp.load(file)
    lam = data['lam']
    spec = data['spec']
    input = {k: data[k] for k in data.files if k not in ['lam', 'spec']}
    return lam, spec, input

class SpectrumDataset(Dataset):
    def __init__(self, files, variable_input=None):
        self.files = files
        self.varied_input = variable_input

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        lam, spec, input = load_spectrum(self.files[idx])
        lam = torch.log10(torch.tensor(lam))  # Log scale the wavelength
        if self.varied_input:
            input = torch.tensor([input[k].item() for k in self.varied_input], dtype=torch.float32)
        else:
            input = torch.tensor([input[k].item() for k in sorted(input.keys())])
        input = torch.tile(input, (spec.shape[0], 1))  # Ensure input shape matches spec
        input = torch.cat([lam[:, None], input], axis=1)  # Concatenate wavelength with parameters
        spec = torch.log10(torch.tensor(spec) + 1e-10)  # Log scale the spectrum
        return spec, input
    
class NormalizeSpectrumDataset(SpectrumDataset):
    def __init__(self, files,
                 spec_mean, spec_std, input_mean, input_std,
                 variable_input=None):
        super().__init__(files, variable_input)
        self.spec_mean = spec_mean
        self.spec_std = spec_std
        self.input_mean = input_mean
        self.input_std =  input_std

    def __getitem__(self, idx):
        spec, input = super().__getitem__(idx)
        spec_mean = torch.tensor(self.spec_mean, dtype=spec.dtype)
        spec_std = torch.tensor(self.spec_std, dtype=spec.dtype)
        input_mean = torch.tensor(self.input_mean, dtype=input.dtype)
        input_std = torch.tensor(self.input_std, dtype=input.dtype)
        
        spec = (spec - spec_mean) / spec_std
        input = (input - input_mean) / input_std
        
        return spec, input

train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
test_files, val_files = train_test_split(test_files, test_size=0.5, random_state=42)
train_dataset = SpectrumDataset(train_files, variable_input=['logzsol', 'tage'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
spec_mean, spec_std, input_mean, input_std = compute_mean_std(train_loader)


train_dataset = NormalizeSpectrumDataset(
    train_files, variable_input=['logzsol', 'tage'],
    spec_mean=spec_mean, spec_std=spec_std,
    input_mean=input_mean, input_std=input_std
)
test_dataset = NormalizeSpectrumDataset(
    test_files, variable_input=['logzsol', 'tage'],
    spec_mean=spec_mean, spec_std=spec_std,
    input_mean=input_mean, input_std=input_std
)
val_dataset = NormalizeSpectrumDataset(
    val_files, variable_input=['logzsol', 'tage'],
    spec_mean=spec_mean, spec_std=spec_std,
    input_mean=input_mean, input_std=input_std
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  

key, subkey = random.split(key)
params = initialise_mlp(
    in_size=3,
    out_size=1,
    hidden_size=64,
    nlayers=3,
    key=subkey,
    scale=1e-1
)

def loss_fn(params, input, spec):
    pred = mlp(params, input)
    return jnp.mean((pred[:, :, 0] - spec) ** 2)

loss = jax.jit(loss_fn)
grad_fn = jax.value_and_grad(loss)


def train_step(params, opt_state, input, spec):
    loss, grads = grad_fn(
        params, input, spec
    )  # Compute loss and gradients
    updates, opt_state = optimizer.update(
        grads, opt_state, params
    )  # Compute updates
    new_params = optax.apply_updates(params, updates)  # Apply updates
    return new_params, opt_state, loss

epochs = 100

pbar = tqdm(range(epochs), desc="Training Progress")

learning_rate = 1e-3
weight_decay = 1e-4

optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
opt_state = optimizer.init(params)

best_loss = float('inf')
patience = 50
tl, vl = [], []
for epoch in pbar:
    trainloss = []
    vlloss = []
    for i, (spec, input) in enumerate(train_loader):
        params, opt_state, train_loss = train_step(
            params, opt_state, jnp.array(input), jnp.array(spec)
        ) 
        trainloss.append(train_loss)
    tl.append(jnp.mean(jnp.array(trainloss)))

    for i, (spec_val, input_val) in enumerate(val_loader):
        val_loss = loss(params, jnp.array(input_val), jnp.array(spec_val))
        vlloss.append(val_loss)
    vl.append(jnp.mean(jnp.array(vlloss)))
    if vl[-1] < best_loss:
        best_loss = vl[-1]
        best_network_params = params
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter > patience:
        break

    pbar.set_postfix(
        {
            "ensemble": epoch + 1,
            "tl": tl[-1],
            "vl": vl[-1],
            "best_loss": best_loss,
        }
    )

print("Training complete.")
print("Best validation loss:", best_loss)
import os
os.makedirs('trained_networks', exist_ok=True)
torch.save(best_network_params, 'trained_networks/best_network_params.pth')
loss = []
for example in test_dataset:
    spec, input = example
    spec = jnp.array(spec)
    input = jnp.array(input)
    pred = mlp(best_network_params, input)
    loss.append(jnp.mean((pred[:, :, 0] - spec) ** 2))

loss = jnp.stack(loss)

plt.hist(loss, bins=50)
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.title('Loss Distribution on Test Set')
plt.savefig('trained_networks/loss_distribution.png')
