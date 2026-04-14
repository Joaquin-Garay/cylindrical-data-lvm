"""Base interface for exponential-family distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np

from ...core.types import Array
from ..base import Distribution


class ExponentialFamily(Distribution, ABC):
    """Abstract base for exponential-family distributions."""

    @property
    @abstractmethod
    def params(self) -> Any:
        """Ordinary parameterization (e.g., mean/covariance)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def natural_param(self) -> Array:
        """Natural parameter vector."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dual_param(self) -> Array:
        """Expectation/dual parameter vector."""
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        x: Array,
        sample_weight: Optional[Array] = None,
        case: str = "classic",
    ) -> "ExponentialFamily":
        """
        Fit model parameters in-place and return self.

        Supported ``case`` values: ``classic``, ``bregman``.
        """
        raise NotImplementedError

    @staticmethod
    def _normalize_weights(weights: Array) -> Array:
        """Validate and normalize weights to sum to one."""
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            raise ValueError("sample_weight must be a 1D array with shape (n,).")
        if w.size == 0:
            raise ValueError("sample_weight must not be empty.")
        if not np.all(np.isfinite(w)):
            raise ValueError("sample_weight contains non-finite values.")
        if np.any(w < 0):
            raise ValueError("sample_weight must be nonnegative.")

        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("sample_weight must sum to a positive value.")
        return w / total

    @staticmethod
    def _validate_case(case: str) -> None:
        if case not in {"classic", "bregman"}:
            raise ValueError("case must be one of {'classic', 'bregman'}.")

    def _input_process(
        self,
        x: Array,
        weights: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """
        Validate inputs and return ``(x, normalized_weights)``.

        - x: shape (n,) or (n, d), finite, n >= 1
        - weights: shape (n,), finite, nonnegative, sum > 0
        """
        x = self._validate_input_samples(x)
        n_samples = int(x.shape[0])
        if weights is None:
            weights = np.full(n_samples, 1.0 / n_samples, dtype=float)
        else:
            weights = self._normalize_weights(weights)
            if weights.shape[0] != n_samples:
                raise ValueError(
                    "sample_weight length mismatch: expected "
                    f"{n_samples}, got {weights.shape[0]}."
                )
        return x, weights
