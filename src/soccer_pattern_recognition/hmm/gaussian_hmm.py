"""Diagonal-covariance Gaussian HMM built from EmissionHMM + GaussianEmission."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.types import Array
from .base import EmissionHMM
from .emissions import GaussianEmission


class GaussianHMM(EmissionHMM):
    """
    Gaussian-emission HMM using the modular emission architecture.

    This class mirrors the common ``GaussianHMM`` workflow while delegating
    emission logic to :class:`GaussianEmission`.
    """

    def __init__(
        self,
        n_components: int = 1,
        *,
        covariance_type: str = "diag",
        min_covar: float = 1e-3,
        means: Array | None = None,
        covars: Array | None = None,
        init_method: str = "kmeans",
        startprob_prior: Any = 1.0,
        transmat_prior: Any = 1.0,
        algorithm: str = "viterbi",
        random_state: Any = None,
        n_iter: int = 10,
        tol: float = 1e-2,
        verbose: bool = False,
        params: str = "stmc",
        init_params: str = "stmc",
        implementation: str = "log",
    ) -> None:
        if covariance_type != "diag":
            raise ValueError(
                "Only 'diag' covariance_type is supported by this GaussianHMM."
            )
        self.covariance_type = covariance_type
        self.min_covar = float(min_covar)
        self.init_method = str(init_method).lower()

        emission = GaussianEmission(
            means=means,
            covars=covars,
            min_covar=min_covar,
            init_method=init_method,
        )
        super().__init__(
            n_components=n_components,
            emission=emission,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
        )

    @property
    def means_(self) -> Array:
        if self.emission.means_ is None:
            raise AttributeError("means_ are not initialized yet.")
        return self.emission.means_

    @means_.setter
    def means_(self, value: Array) -> None:
        self.emission.means_ = np.asarray(value, dtype=float)

    @property
    def covars_(self) -> Array:
        if self.emission.covars_ is None:
            raise AttributeError("covars_ are not initialized yet.")
        return self.emission.covars_

    @covars_.setter
    def covars_(self, value: Array) -> None:
        self.emission.covars_ = np.asarray(value, dtype=float)

    def _check(self):
        if self.covariance_type != "diag":
            raise ValueError(
                "Only 'diag' covariance_type is supported by this GaussianHMM."
            )
        super()._check()
