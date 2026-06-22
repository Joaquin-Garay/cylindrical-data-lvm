"""Custom exponential-family distribution combining Gaussian and Von Mises parts."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...core.types import Array
from ..abstract_cylindrical import AbstractCylindrical
from .base import ExponentialFamily
from .multi_gaussian import MultivariateGaussian
from .vmf import VonMisesFisher

# ----- Independent Gaussian-vonMises distribution -----
class IndCylindrical(AbstractCylindrical, ExponentialFamily):
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

        self._d_gauss = self._validate_positive_int(d_gauss, name="d_gauss", minimum=1)
        self._d_vmf = self._validate_positive_int(d_vmf, name="d_vmf", minimum=2)
        self._vmf = VonMisesFisher(self._d_vmf,
                                   mu=mu_vmf,
                                   kappa=kappa)
        self._gaussian = MultivariateGaussian(self._d_gauss,
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

    def _split_input(self, x: Array) -> Tuple[Array, Array]:
        x = self._validate_input_matrix(x, n_features=self.d_total, name="x")
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

    # Getter in Cylindrical-style signature
    @property
    def cond_cov(self) -> Array:
        return self._gaussian.covariance

    @property
    def unconditional_gauss_cov(self) -> Array:
        return self._gaussian.covariance

    @property
    def mu_gauss(self) -> Array:
        return self._gaussian.mean

    @property
    def cross_cov(self) -> Array:
        return np.zeros(shape=(self._d_gauss, self._d_vmf))

    @property
    def cross_corr(self) -> Array:
        return np.zeros(shape=(self._d_gauss, self._d_vmf))

    @property
    def vmf(self):
        return self._vmf

    @property
    def gaussian(self):
        return self._gaussian

    @property
    def dual_param(self) -> Tuple[Array, Array]:
        return self._gaussian.dual_param, self._vmf.dual_param

    def log_pdf(self, x: Array) -> Array:
        x_gauss, x_vmf = self._split_input(x)
        return self._gaussian.log_pdf(x_gauss) + self._vmf.log_pdf(x_vmf)

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        self._validate_n_samples(n)
        rng = self._resolve_rng(rng)
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

