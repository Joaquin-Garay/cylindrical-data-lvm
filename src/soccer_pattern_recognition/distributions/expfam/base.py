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

    @classmethod
    def _normalize_weights(cls, weights: Array) -> Array:
        """Validate and normalize weights to sum to one."""
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.size == 0:
            raise ValueError("sample_weight must be a non-empty 1D array with shape (n,).")
        return cls._normalize_sample_weight(w, w.shape[0])

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
        weights = self._normalize_sample_weight(weights, n_samples)
        return x, weights
