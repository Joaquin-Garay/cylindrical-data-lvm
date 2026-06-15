"""Base interfaces for probability distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..core.types import Array


class Distribution(ABC):
    """Abstract interface for distribution models."""

    @abstractmethod
    def log_pdf(self, x: Array) -> Array:
        """
        Return log-density values for each sample.

        Accepted input shapes:
        - univariate: (n,)
        - multivariate: (n, d)
        """
        raise NotImplementedError

    def pdf(self, x: Array) -> Array:
        """Default density computed from ``log_pdf``."""
        return np.exp(self.log_pdf(x))

    @abstractmethod
    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        """
        Draw ``n`` i.i.d. samples.

        Output shape:
        - univariate: (n,)
        - multivariate: (n, d)
        """
        raise NotImplementedError

    @staticmethod
    def _validate_n_samples(n: int) -> None:
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be an integer >= 1.")

    @staticmethod
    def _validate_positive_int(value: int, *, name: str, minimum: int = 1) -> int:
        if not isinstance(value, (int, np.integer)) or int(value) < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
        return int(value)

    @staticmethod
    def _resolve_rng(rng: Optional[np.random.RandomState]) -> np.random.RandomState:
        if rng is None:
            return np.random.RandomState()
        if not isinstance(rng, np.random.RandomState):
            raise TypeError("rng must be None or np.random.RandomState.")
        return rng

    @staticmethod
    def _normalize_sample_weight(sample_weight: Optional[Array], n_samples: int) -> Array:
        if sample_weight is None:
            return np.full(n_samples, 1.0 / n_samples, dtype=float)

        w = np.asarray(sample_weight, dtype=float)
        if w.ndim != 1:
            raise ValueError("sample_weight must be a 1D array with shape (n,).")
        if w.shape[0] != n_samples:
            raise ValueError(
                f"sample_weight length mismatch: expected {n_samples}, got {w.shape[0]}."
            )
        if not np.all(np.isfinite(w)):
            raise ValueError("sample_weight contains non-finite values.")
        if np.any(w < 0.0):
            raise ValueError("sample_weight must be nonnegative.")

        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("sample_weight must sum to a positive value.")
        return w / total

    @staticmethod
    def _validate_finite_array(x: Array, *, name: str) -> Array:
        arr = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")
        return arr

    @classmethod
    def _validate_vector(
        cls,
        x: Array,
        *,
        size: int,
        name: str,
    ) -> Array:
        arr = cls._validate_finite_array(x, name=name)
        if arr.ndim != 1 or arr.shape[0] != size:
            raise ValueError(f"{name} must have shape ({size},).")
        return arr

    @classmethod
    def _validate_matrix(
        cls,
        x: Array,
        *,
        shape: tuple[int, int],
        name: str,
        symmetric: bool = False,
    ) -> Array:
        arr = cls._validate_finite_array(x, name=name)
        if arr.ndim != 2 or arr.shape != shape:
            raise ValueError(f"{name} must have shape {shape}.")
        if symmetric and not np.allclose(arr, arr.T):
            raise ValueError(f"{name} must be symmetric.")
        return arr

    @classmethod
    def _validate_input_matrix(
        cls,
        x: Array,
        *,
        n_features: int,
        name: str = "x",
        allow_single_vector: bool = False,
    ) -> Array:
        arr = cls._validate_input_samples(x)
        if allow_single_vector and arr.ndim == 1:
            if arr.shape[0] != n_features:
                raise ValueError(
                    f"{name} expects shape ({n_features},) or (n, {n_features})."
                )
            arr = arr[np.newaxis, :]
        if arr.ndim != 2 or arr.shape[1] != n_features:
            raise ValueError(f"{name} expects shape (n, {n_features}).")
        return arr

    @staticmethod
    def _validate_input_samples(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        if x.ndim not in (1, 2):
            raise ValueError("x must be a 1D (n,) or 2D (n, d) array.")
        if x.shape[0] < 1:
            raise ValueError("x must contain at least one sample.")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non-finite values.")
        return x
