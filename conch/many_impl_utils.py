"""Utilities for comparing multiple different implementations of the same computation."""

from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import torch


def mad(a: Union[torch.Tensor, jax.Array, np.ndarray],
        b: Union[torch.Tensor, jax.Array, np.ndarray]):
    """Returns the maximum absolute difference between two tensors: `(a - b).abs().max()`

    Aka the l-infinity norm.
    """
    def to_numpy(t):
        if isinstance(t, np.ndarray):
            return t
        elif isinstance(t, torch.Tensor):
            return t.cpu().numpy()
        elif isinstance(t, jnp.ndarray):
            return np.array(t)
        else:
            raise ValueError(f"Unknown tensor type: {type(t)}")

    return np.max(np.abs(to_numpy(a) - to_numpy(b)))