"""Training loop for astroemu."""

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from astroemu.dataloaders import SpectrumDataset
from astroemu.losses import mse
from astroemu.network import initialise_mlp, mlp


def train(
    train_dataset: SpectrumDataset,
    val_dataset: SpectrumDataset,
    hidden_size: int,
    nlayers: int,
    act: str = "relu",
    epochs: int = 1000,
    patience: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    key: int = 0,
) -> tuple[dict, list[float], list[float]]:
    """Train an MLP emulator on spectral data using AdamW.

    Initialises an MLP via initialise_mlp, then trains it using batches
    from train_dataset and val_dataset (both must have tiling=True).
    The per-batch train and validation steps are JIT-compiled. Training
    stops early if the validation loss does not improve for patience
    consecutive epochs, and the best parameters are returned.

    Args:
        train_dataset (SpectrumDataset): Training dataset with tiling=True.
        val_dataset (SpectrumDataset): Validation dataset with tiling=True.
        hidden_size (int): Number of nodes in each hidden layer.
        nlayers (int): Number of hidden layers.
        act (str): Activation function name from jax.nn. Defaults to
            "relu".
        epochs (int): Maximum number of training epochs. Defaults to 1000.
        patience (int): Early stopping patience in epochs. Defaults to 50.
        learning_rate (float): AdamW learning rate. Defaults to 1e-3.
        weight_decay (float): AdamW weight decay. Defaults to 1e-4.
        batch_size (int): Number of spectra per batch. Defaults to 32.
        key (int): Integer seed for JAX PRNG. Defaults to 0.

    Returns:
        tuple[dict, list[float], list[float]]: Best network parameters,
            per-epoch training losses, and per-epoch validation losses.
    """
    # Infer input size from the dataset: n_params + 1 (x is prepended
    # to parameters in tiling mode, so in_size = n_params + 1).
    _, _, params_sample = train_dataset[0]
    in_size = int(params_sample.shape[0]) + 1
    out_size = 1

    # Initialise network parameters.
    rng_key = jax.random.PRNGKey(key)
    rng_key, init_key = jax.random.split(rng_key)
    params = initialise_mlp(in_size, out_size, hidden_size, nlayers, init_key)

    # Initialise AdamW optimizer.
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    # JIT-compiled training step. act is a closure variable and is
    # therefore treated as a compile-time constant by JAX.
    @jax.jit
    def train_step(
        params: dict,
        opt_state: optax.OptState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> tuple[dict, optax.OptState, jnp.ndarray]:
        """Compute loss and gradients, then apply an AdamW update.

        Args:
            params (dict): Current network parameters.
            opt_state (optax.OptState): Current optimizer state.
            inputs (jnp.ndarray): Tiled input array of shape
                (batch * len_x, n_params + 1).
            targets (jnp.ndarray): Target values of shape (batch * len_x,).

        Returns:
            tuple[dict, optax.OptState, jnp.ndarray]: Updated parameters,
                updated optimizer state, and scalar batch loss.
        """

        def loss_fn(p: dict) -> jnp.ndarray:
            preds = mlp(p, inputs, act)
            return mse(preds.squeeze(-1), targets)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # JIT-compiled validation step.
    @jax.jit
    def val_step(
        params: dict,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute validation loss without updating parameters.

        Args:
            params (dict): Current network parameters.
            inputs (jnp.ndarray): Tiled input array of shape
                (batch * len_x, n_params + 1).
            targets (jnp.ndarray): Target values of shape (batch * len_x,).

        Returns:
            jnp.ndarray: Scalar batch loss.
        """
        preds = mlp(params, inputs, act)
        return mse(preds.squeeze(-1), targets)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = jnp.inf
    best_params = params
    patience_counter = 0

    pbar = tqdm(range(epochs), desc="Training epochs")

    for epoch in pbar:
        rng_key, train_key = jax.random.split(rng_key)

        # Training pass.
        epoch_train_loss = 0.0
        n_train_batches = 0
        for targets, inputs in train_dataset.get_batch_iterator(
            batch_size, shuffle=True, key=train_key
        ):
            params, opt_state, batch_loss = train_step(
                params, opt_state, inputs, targets
            )
            epoch_train_loss += float(batch_loss)
            n_train_batches += 1
        epoch_train_loss /= n_train_batches

        # Validation pass.
        epoch_val_loss = 0.0
        n_val_batches = 0
        for targets, inputs in val_dataset.get_batch_iterator(
            batch_size, shuffle=False
        ):
            epoch_val_loss += float(val_step(params, inputs, targets))
            n_val_batches += 1
        epoch_val_loss /= n_val_batches

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Early stopping: save best params and reset counter on improvement.
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best val loss: {best_val_loss:.4f})."
                )
                break

        pbar.set_postfix(
            {
                "train_loss": f"{epoch_train_loss:.4f}",
                "val_loss": f"{epoch_val_loss:.4f}",
                "best_val_loss": f"{best_val_loss:.4f}",
            }
        )

    return best_params, train_losses, val_losses
