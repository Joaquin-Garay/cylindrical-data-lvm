"""Shared validation helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def validate_sample_weight(x: np.ndarray, sample_weight: Sequence[float] | None) -> np.ndarray:
    """
    Validate and normalize sample weights.

    Parameters
    ----------
    x : np.ndarray
        Input samples with shape ``(n_samples, ...)``.
    sample_weight : Sequence[float] | None
        Optional nonnegative weights of length ``n_samples``.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Normalized sample weights that sum to 1.

    Raises
    ------
    ValueError
        If shape, finiteness, sign, or normalization constraints are violated.
    """
    n_obs = x.shape[0]
    if sample_weight is None:
        return np.full(n_obs, 1.0 / n_obs, dtype=float)

    w = np.asarray(sample_weight, dtype=float)
    if w.ndim != 1:
        raise ValueError("sample_weight must be a 1D array.")
    if w.shape[0] != n_obs:
        raise ValueError(
            f"sample_weight length mismatch: expected {n_obs}, got {w.shape[0]}."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight contains non-finite values.")
    if np.any(w < 0.0):
        raise ValueError("sample_weight must be nonnegative.")

    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("sample_weight must sum to a positive value.")
    return w / total
