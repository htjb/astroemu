import jax
import jax.numpy as jnp
from jax import random


def initialise_mlp(in_size, out_size, hidden_size, nlayers, key, scale=1e-1):
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


def mlp(params, input):
    act_fn = getattr(jax.nn, "relu")
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
