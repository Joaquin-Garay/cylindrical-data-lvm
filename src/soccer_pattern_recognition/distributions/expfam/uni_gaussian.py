"""Univariate Gaussian exponential-family distribution."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...core.types import Array
from .base import ExponentialFamily


class UnivariateGaussian(ExponentialFamily):
    """
    N(mean, variance) in natural form:
        theta = ( -mu/sigma^2, -1/(2*sigma^2) )
    dual / mean-value form:
        eta =   ( mu , mu^2 + sigma^2 )
    """

    def __init__(self, mean: float = 0.0, variance: float = 1.0):
        self._mean = float(mean)
        self._variance = float(variance)
        self._natural_param: Optional[Array] = None
        self._dual_param: Optional[Array] = None
        self._validate()
        self._update_params()

    def _validate(self) -> None:
        if self._variance <= 0:
            raise ValueError("variance must be positive.")
        if self._natural_param is not None and self._natural_param[1] >= 0:
            raise ValueError("Second natural parameter must be negative.")
        if self._dual_param is not None and self._dual_param[1] <= self._dual_param[0] ** 2:
            raise ValueError("eta2 - eta1^2 must be positive.")

    def _update_params(self) -> None:
        self._natural_param = np.array([
            -self._mean / self._variance,
            -1.0 / (2.0 * self._variance)
        ])
        self._dual_param = np.array([
            self._mean,
            self._mean ** 2 + self._variance
        ])

    # ---- Getters and Setters ----
    @property
    def params(self) -> Tuple[float, float]:
        return self._mean, self._variance

    @params.setter
    def params(self, value: Tuple[float, float]):
        self._mean, self._variance = value
        self._validate()
        self._update_params()

    @property
    def natural_param(self):
        return self._natural_param.copy()

    @natural_param.setter
    def natural_param(self, theta: Array):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2,):
            raise ValueError("natural_param must be a length-2 vector.")
        self._natural_param = theta
        self._mean = -0.5 * theta[0] / theta[1]
        self._variance = -0.5 / theta[1]
        self._validate()
        self._update_params()

    @property
    def dual_param(self):
        return self._dual_param.copy()

    @dual_param.setter
    def dual_param(self, eta: Array) -> None:
        eta = np.asarray(eta, dtype=float)
        if eta.shape != (2,):
            raise ValueError("dual_param must be a length-2 vector.")
        self._dual_param = eta
        self._mean = eta[0]
        self._variance = eta[1] - eta[0] ** 2
        self._validate()
        self._update_params()

    @staticmethod
    def from_dual_to_ordinary(eta: Array) -> Tuple[float, float]:
        return float(eta[0]), float(eta[1] - eta[0] ** 2)

    # ---- densities ----
    def log_pdf(self, x: Array) -> Array:
        x = self._validate_input_samples(x)
        if x.ndim == 2:
            x = self._validate_input_matrix(x, n_features=1, name="x")[:, 0]
        return -0.5 * ((x - self._mean) ** 2) / self._variance - 0.5 * np.log(2 * np.pi * self._variance)

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        self._validate_n_samples(n)
        rng = self._resolve_rng(rng)
        return rng.normal(loc=self._mean, scale=np.sqrt(self._variance), size=n)

    # pdf inherited from base

    # ---- Calibration ----
    def fit(self,
            x: Array,
            sample_weight: Optional[Array] = None,
            case: str = "classic",
            ) -> "UnivariateGaussian":

        self._validate_case(case)
        x, sample_weight = self._input_process(x, sample_weight)
        if x.ndim == 2:
            x = self._validate_input_matrix(x, n_features=1, name="x")[:, 0]
        match case:
            case "bregman":
                # compute dual/expectation parameters using sufficient statistics.
                eta = np.array([np.average(x, weights=sample_weight),
                                np.average(x ** 2, weights=sample_weight)])
                self.dual_param = eta
            case _:
                mu = np.average(x, weights=sample_weight)
                diff = x - mu
                variance = np.inner(sample_weight * diff, diff)

                self._mean = mu
                self._variance = variance
                self._validate()
                self._update_params()
        return self

    def __repr__(self) -> str:
        return f"UnivariateGaussian(mean={self._mean:.3f}, variance={self._variance:.3f})"
