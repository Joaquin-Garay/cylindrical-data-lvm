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
    def _validate_input_samples(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        if x.ndim not in (1, 2):
            raise ValueError("x must be a 1D (n,) or 2D (n, d) array.")
        if x.shape[0] < 1:
            raise ValueError("x must contain at least one sample.")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non-finite values.")
        return x
