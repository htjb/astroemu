"""Neural network implementations for emu package."""

import jax
import jax.numpy as jnp
from jax import random


def initialise_mlp(
    in_size: int,
    out_size: int,
    hidden_size: int,
    nlayers: int,
    key: int,
    scale: float = 1e-1,
) -> dict:
    """Initialize MLP parameters.

    Args:
        in_size (int): Input size.
        out_size (int): Output size.
        hidden_size (int): Hidden layer size.
        nlayers (int): Number of hidden layers.
        key (int): JAX random key.
        scale (float, optional): Scale for weight initialization.
            Defaults to 1e-1.

    Returns:
        dict: MLP parameters.
    """
    keys = random.split(key, nlayers * 2 + 2 + 2)
    weights = (
        [
            {
                "weights" + str(0): scale
                * random.normal(keys[0], (in_size, hidden_size)),
                "bias" + str(0): scale
                * random.normal(keys[1], (hidden_size,)),
            }
        ]
        + [
            {
                "weights" + str(i + 1): scale
                * random.normal(keys[i + 2], (hidden_size, hidden_size)),
                "bias" + str(i + 1): scale
                * random.normal(keys[i + 3], (hidden_size,)),
            }
            for i in range(nlayers)
        ]
        + [
            {
                "weights" + str(nlayers + 1): scale
                * random.normal(keys[-2], (hidden_size, out_size)),
                "bias" + str(nlayers + 1): scale
                * random.normal(keys[-1], (out_size,)),
            }
        ]
    )
    return {k: v for d in weights for k, v in d.items()}


def mlp(params: dict, input: jnp.ndarray, act: str = "relu") -> jnp.ndarray:
    """Multi-layer perceptron with residual connections.

    Args:
        params (dict): MLP parameters.
        input (jnp.ndarray): Input array of shape [..., in_size].
        act (str): Activation function name from jax.nn. Defaults to
            "relu". Must be treated as a static argument if JIT-compiling
            mlp directly (static_argnames=("act",)).

    Returns:
        jnp.ndarray: Output array of shape [..., out_size].
    """
    act_fn = getattr(jax.nn, act)
    num_layers = len(params) // 2  # total layers: input + hidden(s) + output

    x = jnp.dot(input, params["weights0"]) + params["bias0"]

    for i in range(1, num_layers - 1):  # exclude final output layer
        residual = x
        x = act_fn(x)
        x = jnp.dot(x, params[f"weights{i}"]) + params[f"bias{i}"]
        # Residual connection (only if shapes match)
        x += residual

    # Final layer: linear only, no activation
    output = (
        jnp.dot(x, params[f"weights{num_layers - 1}"])
        + params[f"bias{num_layers - 1}"]
    )
    return output
