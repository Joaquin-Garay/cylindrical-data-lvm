"""Categorical distribution."""

from __future__ import annotations

from typing import Optional
import numpy as np

from ...core.types import Array
from ..base import Distribution


class Categorical(Distribution):
    """
    Categorical distribution over ``k`` classes.

    Parameters
    - probs: class probabilities of shape (k,). Values must be nonnegative
      and sum to a positive value. They are normalized to sum to 1.
    """

    def __init__(self, probs: Array):
        self._probs = self._validate_probs(probs)

    @classmethod
    def _validate_probs(cls, probs: Array) -> Array:
        p = cls._validate_finite_array(probs, name="probs")
        if p.ndim != 1:
            raise ValueError("probs must be a 1D array with shape (k,).")
        if p.size < 2:
            raise ValueError("probs must contain at least two categories.")
        if np.any(p < 0):
            raise ValueError("probs must be nonnegative.")
        total = float(p.sum())
        if total <= 0.0:
            raise ValueError("probs must sum to a positive value.")
        return p / total

    @property
    def probs(self) -> Array:
        return self._probs.copy()

    @probs.setter
    def probs(self, value: Array) -> None:
        self._probs = self._validate_probs(value)

    @property
    def n_categories(self) -> int:
        return int(self._probs.size)

    def log_pdf(self, x: Array) -> Array:
        x = self._validate_input_samples(x)

        # Case 1: class indices with shape (n,)
        if x.ndim == 1:
            if not np.all(np.equal(x, np.round(x))):
                raise ValueError("Categorical labels must be integer-valued.")
            idx = x.astype(int)
            if np.any(idx < 0) or np.any(idx >= self.n_categories):
                raise ValueError(
                    f"Category indices must be in [0, {self.n_categories - 1}]."
                )
            return np.log(self._probs[idx])

        # Case 2: one-hot encoded rows with shape (n, k)
        if x.ndim == 2:
            if x.shape[1] != self.n_categories:
                raise ValueError(
                    f"One-hot x must have {self.n_categories} columns."
                )
            if not np.all((x == 0) | (x == 1)):
                raise ValueError("One-hot x must contain only 0/1 values.")
            row_sums = x.sum(axis=1)
            if not np.all(row_sums == 1):
                raise ValueError("Each one-hot row must sum to 1.")
            return x @ np.log(self._probs)

        raise ValueError("x must be a 1D label array or 2D one-hot array.")

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        self._validate_n_samples(n)
        rng = self._resolve_rng(rng)
        return rng.choice(self.n_categories, size=n, p=self._probs)

    def __repr__(self) -> str:
        probs_str = np.array2string(self._probs, precision=4, separator=", ")
        return f"Categorical(k={self.n_categories}, probs={probs_str})"
