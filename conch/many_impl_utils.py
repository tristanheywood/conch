"""Utilities for comparing multiple different implementations of the same computation."""

from functools import cache, cached_property
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch

AnyTensor = Union[torch.Tensor, jax.Array, np.ndarray]


def tensor_to_numpy(t: AnyTensor):
    if isinstance(t, np.ndarray):
        return t
    elif isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    elif isinstance(t, jnp.ndarray):
        return np.array(t)
    else:
        raise ValueError(f"Unknown tensor type: {type(t)}")


def mad(a: AnyTensor, b: AnyTensor):
    """Returns the maximum absolute difference between two tensors: `(a - b).abs().max()`

    Aka the l-infinity norm.
    """
    return np.max(np.abs(tensor_to_numpy(a) - tensor_to_numpy(b)))


class TensorComparison:
    """
    Class for comparing two tensors, assumed to be the results of the same computation
    implemented in two different ways.

    The motivation for using this class instead of something like `np.isclose`, is that
    measurements of relative error are not necessarily informative. Due to the linear
    structure of neural networks, large relative errors in values which are close to
    zero typically have minimal impact on overall results. At the same time, measures
    of absolute error are difficult to interpret without knowing the scale of the the
    tensors. For these reasons, `TensorComparison` provides measures which combine
    both absolute and relative error.
    """
    def __init__(self, ref: AnyTensor, other: AnyTensor):
        self.ref = ref
        self.other = other

        self.ref_np = tensor_to_numpy(ref)
        self.other_np = tensor_to_numpy(other)

        if self.ref_np.shape != self.other_np.shape:
            raise ValueError(
                f"Shapes of reference and other tensors do not match: "
                f"{self.ref_np.shape} vs {self.other_np.shape}")

    @cached_property
    def diff(self) -> np.ndarray:
        return self.ref_np - self.other_np

    @cached_property
    def abs_diff(self) -> np.ndarray:
        return np.abs(self.diff)

    @cache
    def mean_abs_diff(self, axis=None) -> np.ndarray:
        return np.mean(self.abs_diff, axis=axis)

    @cached_property
    def mad(self) -> float:
        return np.max(self.abs_diff)

    @cached_property
    def rel_mad(self) -> float:
        """Size of the maximum absolute difference relative to the median absolute value
        of the reference tensor."""
        return self.mad / np.median(np.abs(self.ref_np))

    @cached_property
    def row_max_rel_err(self) -> float:
        assert len(self.ref_np.shape) == 2

        row_sums = np.sum(np.abs(self.ref_np), axis=1)
        row_mean_abs_diff = self.mean_abs_diff(axis=1)
        row_rel_err = row_mean_abs_diff / row_sums
        return np.max(row_rel_err)

    @cached_property
    def col_max_rel_err(self) -> float:
        assert len(self.ref_np.shape) == 2

        col_sums = np.sum(np.abs(self.ref_np), axis=0)
        col_mean_abs_diff = self.mean_abs_diff(axis=0)
        col_rel_err = col_mean_abs_diff / col_sums
        return np.max(col_rel_err)

    def describe(self) -> pd.DataFrame:
        """Returns a DataFrame with a summary of the comparison results."""
        data = {
            "shape": [self.ref_np.shape],
            "mad": self.mad,
            "rel_mad": self.rel_mad,
            "row_max_rel_err": self.row_max_rel_err,
            "col_max_rel_err": self.col_max_rel_err,
        }

        return pd.DataFrame(data).T