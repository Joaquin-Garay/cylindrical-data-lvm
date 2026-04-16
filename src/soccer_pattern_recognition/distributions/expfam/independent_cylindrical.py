"""Custom exponential-family distribution combining Gaussian and Von Mises parts."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...core.types import Array
from .base import ExponentialFamily
from .gaussian import MultivariateGaussian
from .vmf import VonMisesFisher

# ----- Independent Gaussian-vonMises distribution -----
class IndCylindrical(ExponentialFamily):
    """
    p(x_1,x_2) = Gauss(x_1) * vMF(x_2)
    """

    def __init__(self,
                 d_gauss: int,
                 d_vmf: int,
                 *,
                 mu_gauss: Optional[Array] = None,
                 cov_gauss: Optional[Array] = None,
                 mu_vmf: Optional[Array] = None,
                 kappa: Optional[float] = None):

        if not isinstance(d_gauss, (int, np.integer)) or int(d_gauss) < 1:
            raise ValueError("d_gauss must be an integer >= 1.")
        if not isinstance(d_vmf, (int, np.integer)) or int(d_vmf) < 2:
            raise ValueError("d_vmf must be an integer >= 2.")

        self._d_gauss = int(d_gauss)
        self._d_vmf = int(d_vmf)
        self._vmf = VonMisesFisher(d_vmf,
                                   mu=mu_vmf,
                                   kappa=kappa)
        self._gaussian = MultivariateGaussian(d_gauss,
                                              mean=mu_gauss,
                                              covariance=cov_gauss)
        self._validate_components()

    def _validate_components(self) -> None:
        if self._gaussian.d != self._d_gauss:
            raise ValueError(
                f"Gaussian dimension mismatch: expected {self._d_gauss}, got {self._gaussian.d}."
            )
        if self._vmf.d != self._d_vmf:
            raise ValueError(
                f"VonMisesFisher dimension mismatch: expected {self._d_vmf}, got {self._vmf.d}."
            )

    @staticmethod
    def _validate_rng(rng: Optional[np.random.RandomState]) -> np.random.RandomState:
        if rng is None:
            return np.random.RandomState()
        if not isinstance(rng, np.random.RandomState):
            raise TypeError("rng must be None or np.random.RandomState.")
        return rng

    def _split_input(self, x: Array) -> Tuple[Array, Array]:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.d_total:
            raise ValueError(
                f"IndCylindrical expects x with shape (n, {self.d_total})."
            )
        return x[:, :self._d_gauss], x[:, self._d_gauss:]

    @property
    def d_gauss(self) -> int:
        return self._d_gauss

    @property
    def d_vmf(self) -> int:
        return self._d_vmf

    @property
    def d_total(self) -> int:
        return self._d_gauss + self._d_vmf

    @property
    def params(self) -> Tuple[Tuple[Array, Array], Tuple[Array, float]]:
        return self._gaussian.params, self._vmf.params

    @params.setter
    def params(self, value: Tuple[Tuple[Array, Array], Tuple[Array, float]]) -> None:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("params must be a pair: ((mean, covariance), (mu, kappa)).")
        gaussian_params, vmf_params = value
        self._gaussian.params = gaussian_params
        self._vmf.params = vmf_params
        self._validate_components()

    @property
    def natural_param(self) -> Tuple[Array, Array]:
        return (
            self._gaussian.natural_param,
            self._vmf.natural_param,
        )

    @property
    def dual_param(self) -> Tuple[Array, Array]:
        return self._gaussian.dual_param, self._vmf.dual_param

    def log_pdf(self, x: Array) -> Array:
        x = self._validate_input_samples(x)
        x_gauss, x_vmf = self._split_input(x)
        return self._gaussian.log_pdf(x_gauss) + self._vmf.log_pdf(x_vmf)

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        self._validate_n_samples(n)
        rng = self._validate_rng(rng)
        x_gauss = np.asarray(self._gaussian.sample(n, rng), dtype=float)
        x_vmf = np.asarray(self._vmf.sample(n, rng), dtype=float)

        return np.concatenate((x_gauss, x_vmf), axis=1)

    def fit(self,
            x: Array,
            sample_weight: Optional[Array] = None,
            case: str = "bregman",
            ) -> "IndCylindrical":

        self._validate_case(case)
        x, sample_weight = self._input_process(x, sample_weight)
        x_gauss, x_vmf = self._split_input(x)
        self._gaussian.fit(x_gauss, sample_weight=sample_weight, case=case)
        self._vmf.fit(x_vmf, sample_weight=sample_weight, case=case)
        self._validate_components()

        return self

    @property
    def gaussian(self) -> MultivariateGaussian:
        return self._gaussian

    @property
    def vmf(self) -> VonMisesFisher:
        return self._vmf

    @property
    def vonmises(self) -> VonMisesFisher:
        return self._vmf
