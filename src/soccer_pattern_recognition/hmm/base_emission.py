"""Emission interfaces and implementations for modular HMMs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

Array = np.ndarray


class BaseEmission(ABC):
    """
    Abstract emission interface expected by ``EmissionHMM``.

    Subclasses must define:
    - which emission parameter symbols they own (``param_symbols``)
    - their own sufficient statistics layout
    - how those statistics are accumulated and converted into M-step updates
    """

    param_symbols: str = "e"

    def __init__(self) -> None:
        self.n_components: int | None = None
        self.n_features: int | None = None

    def bind(self, n_components: int, n_features: int) -> None:
        """Bind HMM dimensions to this emission object."""
        if n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if n_features < 1:
            raise ValueError("n_features must be >= 1.")
        self.n_components = int(n_components)
        self.n_features = int(n_features)

    def _require_binding(self) -> tuple[int, int]:
        if self.n_components is None or self.n_features is None:
            raise RuntimeError(
                "Emission is not bound. Call bind(n_components, n_features) first."
            )
        return self.n_components, self.n_features

    @abstractmethod
    def get_n_fit_scalars_per_param(self) -> Mapping[str, int]:
        """Return free scalar counts per owned parameter symbol."""
        raise NotImplementedError

    @abstractmethod
    def initialize(self, x: Array, init_params: str, random_state: Any) -> None:
        """Initialize emission parameters before EM iterations."""
        raise NotImplementedError

    @abstractmethod
    def check(self) -> None:
        """Validate emission parameters."""
        raise NotImplementedError

    @abstractmethod
    def compute_log_likelihood(self, x: Array) -> Array:
        """Return log p(x_t | z_t=k) matrix with shape (n_samples, n_components)."""
        raise NotImplementedError

    @abstractmethod
    def initialize_sufficient_statistics(self) -> dict[str, Any]:
        """Allocate emission-specific sufficient statistics."""
        raise NotImplementedError

    @abstractmethod
    def accumulate_sufficient_statistics(
        self,
        stats: dict[str, Any],
        x: Array,
        posteriors: Array,
        params: str,
    ) -> None:
        """Accumulate emission-specific sufficient statistics."""
        raise NotImplementedError

    @abstractmethod
    def m_step(self, stats: dict[str, Any], params: str) -> None:
        """Update emission parameters from accumulated sufficient statistics."""
        raise NotImplementedError

    @abstractmethod
    def sample_from_state(self, state: int, random_state: Any) -> Array:
        """Sample one observation vector from a specific hidden state."""
        raise NotImplementedError