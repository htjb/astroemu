from emu.network import initialise_mlp, mlp
from emu.utils import compute_mean_std
import jax
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from emu.normalisation import standardise, log_base_10
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

train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
test_files, val_files = train_test_split(test_files, test_size=0.5, random_state=42)

super_forward_pipeline = log_base_10(xselector=[0, 2])

train_dataset = SpectrumDataset(train_files, 'lam', 'spec', 
                                forward_pipeline=super_forward_pipeline,
                                variable_input=['logzsol', 'tage'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

spec_mean, spec_std, input_mean, input_std = compute_mean_std(train_loader)

standardiser = standardise(
    y_mean=spec_mean,
    y_std=spec_std,
    x_mean=input_mean,
    x_std=input_std
)

train_dataset = NormalizeSpectrumDataset(
    train_files, 'lam', 'spec',
    super_forward_pipeline=super_forward_pipeline,
    forward_pipeline=standardiser,
    variable_input=['logzsol', 'tage']
)
test_dataset = NormalizeSpectrumDataset(
    test_files, 'lam', 'spec',
    super_forward_pipeline=super_forward_pipeline,
    forward_pipeline=standardiser,
    variable_input=['logzsol', 'tage']
)
val_dataset = NormalizeSpectrumDataset(
    val_files, 'lam', 'spec',
    super_forward_pipeline=super_forward_pipeline,
    forward_pipeline=standardiser,
    variable_input=['logzsol', 'tage']
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

epochs = 2

pbar = tqdm(range(epochs), desc="Training Progress")

learning_rate = 1e-3
weight_decay = 1e-4

optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
opt_state = optimizer.init(params)

best_loss = float('inf')
patience = 50
patience_counter = 0
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
count = 0
for example in test_loader:
    if count < 100:
        spec, input = example
        input = jnp.array(input)
        pred = mlp(best_network_params, input)
        pred = torch.tensor(pred[:, :, 0])
        spec, _ = standardiser.backward(spec, None)
        pred, _ = standardiser.backward(pred, None)
        spec, _ = super_forward_pipeline.backward(spec, None)
        pred, _ = super_forward_pipeline.backward(pred, None)
        plt.scatter(pred.numpy(), spec.numpy(), c='k', s=1, alpha=0.5)
        loss.append(torch.abs((pred - spec)/ (spec))*100)  # Avoid division by zero
        count += 1
plt.plot(plt.xlim(), plt.ylim(), 'r--')  # Diagonal line
plt.loglog()
plt.xlabel('Predicted Spectrum')
plt.ylabel('Actual Spectrum')
plt.title('Predicted vs Actual Spectrum')
plt.savefig(f'trained_networks/predicted_vs_actual_{count}.png')
plt.close()

loss = torch.stack(loss)
mask = torch.isfinite(loss)
loss = loss[mask]
print("Test Loss:", torch.mean(loss))

plt.hist(loss.flatten(), bins=50)
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.title('Loss Distribution on Test Set')
plt.savefig('trained_networks/loss_distribution.png')


for example in test_loader:
    spec, input = example
    input = jnp.array(input)
    pred = mlp(best_network_params, input)
    print(pred.shape, spec.shape)
    print(input.shape)
    pred = torch.tensor(pred[:, :, 0])
    spec, _ = standardiser.backward(spec, None)
    pred, _ = standardiser.backward(pred, None)
    spec, _ = super_forward_pipeline.backward(spec, None)
    pred, _ = super_forward_pipeline.backward(pred, None)
    
    plt.figure(figsize=(10, 5))
    plt.plot(spec.numpy(), label='Actual Spectrum', color='blue')
    plt.plot(pred.numpy(), label='Predicted Spectrum', color='orange')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Spectrum Value')
    plt.title('Actual vs Predicted Spectrum')
    plt.legend()
    plt.savefig(f'trained_networks/spectrum_comparison_example.png')
    plt.close()
    exit()