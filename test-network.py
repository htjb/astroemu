"""Test script for training and evaluating the neural network."""

import glob
import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import torch
from jax import random
from optax import OptState
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from emu.dataloaders import NormalizeSpectrumDataset, SpectrumDataset
from emu.network import initialise_mlp, mlp
from emu.normalisation import log_base_10, standardise
from emu.utils import compute_mean_std

key = random.PRNGKey(0)

base_dir = "fsps-data1"
files = glob.glob(
    base_dir + "/*.npz"
)  # [:1000]  # Limit to 1000 files for testing

train_files, test_files = train_test_split(
    files, test_size=0.2, random_state=42
)
test_files, val_files = train_test_split(
    test_files, test_size=0.5, random_state=42
)

super_forward_pipeline = log_base_10(xselector=[0, 2])

train_dataset = SpectrumDataset(
    train_files,
    "lam",
    "spec",
    forward_pipeline=super_forward_pipeline,
    variable_input=["logzsol", "tage"],
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

spec_mean, spec_std, input_mean, input_std = compute_mean_std(train_loader)

standardiser = standardise(
    y_mean=spec_mean, y_std=spec_std, x_mean=input_mean, x_std=input_std
)

train_dataset = NormalizeSpectrumDataset(
    train_files,
    "lam",
    "spec",
    super_forward_pipeline=super_forward_pipeline,
    forward_pipeline=standardiser,
    variable_input=["logzsol", "tage"],
)
test_dataset = NormalizeSpectrumDataset(
    test_files,
    "lam",
    "spec",
    super_forward_pipeline=super_forward_pipeline,
    forward_pipeline=standardiser,
    variable_input=["logzsol", "tage"],
)
val_dataset = NormalizeSpectrumDataset(
    val_files,
    "lam",
    "spec",
    super_forward_pipeline=super_forward_pipeline,
    forward_pipeline=standardiser,
    variable_input=["logzsol", "tage"],
)

hyperparameters = {
    "in_size": 3,  # wavelength + logzsol + tage
    "out_size": 1,  # spectrum value
    "hidden_size": 64,
    "nlayers": 3,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
}

train_loader = DataLoader(
    train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=hyperparameters["batch_size"], shuffle=False
)
val_loader = DataLoader(
    val_dataset, batch_size=hyperparameters["batch_size"], shuffle=False
)

key, subkey = random.split(key)
params = initialise_mlp(
    in_size=hyperparameters["in_size"],
    out_size=hyperparameters["out_size"],
    hidden_size=hyperparameters["hidden_size"],
    nlayers=hyperparameters["nlayers"],
    key=subkey,
)


def loss_fn(
    params: dict, input: jnp.ndarray, spec: jnp.ndarray
) -> jnp.ndarray:
    """Compute mean squared error loss.

    Args:
        params (dict): MLP parameters.
        input (jnp.ndarray): Input array.
        spec (jnp.ndarray): True spectrum array.

    Returns:
        jnp.ndarray: Mean squared error loss.
    """
    pred = mlp(params, input)
    return jnp.mean((pred[:, :, 0] - spec) ** 2)


loss = jax.jit(loss_fn)
grad_fn = jax.value_and_grad(loss)


def train_step(
    params: dict, opt_state: OptState, input: jnp.ndarray, spec: jnp.ndarray
) -> tuple[dict, OptState, jnp.ndarray]:
    """Perform a single training step.

    Args:
        params (dict): MLP parameters.
        opt_state (OptState): Optimizer state.
        input (jnp.ndarray): Input array.
        spec (jnp.ndarray): True spectrum array.

    Returns:
        tuple[dict, any, jnp.ndarray]: Updated parameters, optimizer state,
            and loss value.
    """
    loss, grads = grad_fn(params, input, spec)  # Compute loss and gradients
    updates, opt_state = optimizer.update(
        grads, opt_state, params
    )  # Compute updates
    new_params = optax.apply_updates(params, updates)  # Apply updates
    return new_params, opt_state, loss


epochs = 1000

pbar = tqdm(range(epochs), desc="Training Progress")

optimizer = optax.adamw(
    learning_rate=hyperparameters["learning_rate"],
    weight_decay=hyperparameters["weight_decay"],
)
opt_state = optimizer.init(params)

best_loss = float("inf")
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

os.makedirs("trained_networks", exist_ok=True)


output = {
    "network_params": best_network_params,
    "hyperparameters": hyperparameters,
    "super_forward_pipeline": super_forward_pipeline,
    "standardiser": standardiser,
}
with open("trained_networks/network_params.pkl", "wb") as f:
    pickle.dump(output, f)

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
        plt.scatter(pred.numpy(), spec.numpy(), c="k", s=1, alpha=0.5)
        loss.append(
            (torch.abs((pred - spec) / (spec)) * 100).flatten()
        )  # Avoid division by zero
        count += 1
plt.plot(plt.xlim(), plt.ylim(), "r--")  # Diagonal line
plt.loglog()
plt.xlabel("Predicted Spectrum")
plt.ylabel("Actual Spectrum")
plt.title("Predicted vs Actual Spectrum")
plt.savefig(f"trained_networks/predicted_vs_actual_{count}.png")
plt.close()

loss = torch.concat(loss)
mask = torch.isfinite(loss)
loss = loss[mask]
print("Test Loss:", torch.mean(loss))

plt.hist(loss.flatten(), bins=50)
plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.title("Loss Distribution on Test Set")
plt.savefig("trained_networks/loss_distribution.png")


spec, input = next(iter(test_loader))
pred = mlp(best_network_params, jnp.array(input))
print(pred.shape, spec.shape)
print(input.shape)
pred = torch.tensor(pred[:, :, 0])
spec, input = standardiser.backward(spec, input)
pred, _ = standardiser.backward(pred, _)
spec, input = super_forward_pipeline.backward(spec, input)
pred, _ = super_forward_pipeline.backward(pred, _)

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
biggest_diff = 0
for j in range(len(spec)):
    if j < 10:
        difference = torch.abs((spec[j] - pred[j]) / spec[j]) * 100
        mask = torch.isfinite(difference)
        difference = difference[mask]
        axes[0].plot(input[0, :, 0], spec[j].numpy(), label=f"Spectrum {j}")
        axes[0].plot(
            input[0, :, 0],
            pred[j].numpy(),
            label=f"Predicted {j}",
            linestyle="--",
        )
        axes[1].plot(input[0, :, 0][mask], difference, label=f"Difference {j}")
        max_diff = torch.max(difference[input[0, :, 0][mask] < 1e6])
        if max_diff > biggest_diff:
            biggest_diff = max_diff
[axes[i].set_xlim(1e2, 1e6) for i in range(2)]
axes[1].set_ylim(0, biggest_diff * 1.1)
axes[1].set_xscale("log")
axes[0].loglog()
axes[0].set_xlabel("Wavelength Index")
axes[0].set_ylabel("Spectrum Value")
plt.savefig("trained_networks/spectrum_comparison_example.png")
plt.close()
