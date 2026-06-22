"""Shared interface for cylindrical distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..core.types import Array


class AbstractCylindrical(ABC):
    """Abstract parameter interface shared by cylindrical distributions."""

    @property
    @abstractmethod
    def d_gauss(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def d_vmf(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def mu_gauss(self) -> Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def cross_cov(self) -> Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def cond_cov(self) -> Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def vmf(self) -> Any:
        raise NotImplementedError

    @staticmethod
    def _validate_comparable(
        cyl1: "AbstractCylindrical",
        cyl2: "AbstractCylindrical",
    ) -> None:
        if not isinstance(cyl1, AbstractCylindrical) or not isinstance(
            cyl2,
            AbstractCylindrical,
        ):
            raise TypeError("cyl1 and cyl2 must be AbstractCylindrical instances.")
        if (cyl1.d_gauss, cyl1.d_vmf) != (cyl2.d_gauss, cyl2.d_vmf):
            raise ValueError(
                "cyl1 and cyl2 must have the same dimensions; "
                f"got ({cyl1.d_gauss}, {cyl1.d_vmf}) and "
                f"({cyl2.d_gauss}, {cyl2.d_vmf})."
            )

    @staticmethod
    def cond_cov_diff(cyl1: "AbstractCylindrical", cyl2: "AbstractCylindrical") -> float:
        AbstractCylindrical._validate_comparable(cyl1, cyl2)
        return float(np.linalg.norm(cyl1.cond_cov - cyl2.cond_cov, ord="fro"))

    @staticmethod
    def cross_cov_diff(cyl1: "AbstractCylindrical", cyl2: "AbstractCylindrical") -> float:
        AbstractCylindrical._validate_comparable(cyl1, cyl2)
        return float(np.linalg.norm(cyl1.cross_cov - cyl2.cross_cov, ord="fro"))

    @staticmethod
    def mu_gauss_diff(cyl1: "AbstractCylindrical", cyl2: "AbstractCylindrical") -> float:
        AbstractCylindrical._validate_comparable(cyl1, cyl2)
        return float(np.linalg.norm(cyl1.mu_gauss - cyl2.mu_gauss))

    @staticmethod
    def kappa_diff(cyl1: "AbstractCylindrical", cyl2: "AbstractCylindrical") -> float:
        AbstractCylindrical._validate_comparable(cyl1, cyl2)
        return float(abs(cyl1.vmf.kappa - cyl2.vmf.kappa))

    @staticmethod
    def mu_vmf_diff(cyl1: "AbstractCylindrical", cyl2: "AbstractCylindrical") -> float:
        """Return angular distance in radians, from 0 to pi."""
        AbstractCylindrical._validate_comparable(cyl1, cyl2)
        inner_product = np.inner(cyl1.vmf.mu, cyl2.vmf.mu)
        return float(np.arccos(np.clip(inner_product, -1.0, 1.0)))

    @staticmethod
    def params_diff(cyl1: "AbstractCylindrical", cyl2: "AbstractCylindrical") -> float:
        AbstractCylindrical._validate_comparable(cyl1, cyl2)
        cond_cov = float(np.linalg.norm(cyl1.cond_cov - cyl2.cond_cov, ord="fro"))
        cross_cov = float(np.linalg.norm(cyl1.cross_cov - cyl2.cross_cov, ord="fro"))
        mu_gauss = float(np.linalg.norm(cyl1.mu_gauss - cyl2.mu_gauss))
        kappa = float(abs(cyl1.vmf.kappa - cyl2.vmf.kappa))
        inner_product = np.inner(cyl1.vmf.mu, cyl2.vmf.mu)
        mu_vmf = float(np.arccos(np.clip(inner_product, -1.0, 1.0)))
        return cond_cov + cross_cov + mu_gauss + kappa + mu_vmf
