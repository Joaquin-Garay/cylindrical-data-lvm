"""Custom exponential-family distribution combining Gaussian and Von Mises parts."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.special import i0e

from ...core.types import Array
from .base import ExponentialFamily
from .gaussian import MultivariateGaussian
from .vonmises import VonMises

# ----- Independent Gaussian-vonMises distribution -----
class IndGaussVM(ExponentialFamily):
    """
    p(x) = Gauss(x|theta_gaus)^alpha * VonMises(x|theta_vm)^beta * C
    """

    def __init__(self, coef_gaus: float, coef_vm: float):
        self._vonmises = VonMises()
        self._gaussian = MultivariateGaussian(d=2)
        self._coef_gauss = float(coef_gaus)
        self._coef_vm = float(coef_vm)
        self._validate_coefs()

    def _validate_coefs(self) -> None:
        if not np.isfinite(self._coef_gauss) or not np.isfinite(self._coef_vm):
            raise ValueError("coef_gaus and coef_vm must be finite.")
        if self._coef_gauss <= 0.0 or self._coef_vm <= 0.0:
            raise ValueError("coef_gaus and coef_vm must be strictly positive.")

    @staticmethod
    def _split_input(x: Array) -> Tuple[Array, Array]:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError(
                "IndGaussVM expects x with shape (n, 4): "
                "[x1, x2, cos(theta), sin(theta)]."
            )
        return x[:, :2], x[:, 2:]

    @property
    def params(self) -> Tuple[Tuple[Array, Array], Tuple[float, float], float, float]:
        return self._gaussian.params, self._vonmises.params, self._coef_gauss, self._coef_vm

    @property
    def natural_param(self) -> Tuple[Array, Array, float, float]:
        return (
            self._gaussian.natural_param,
            self._vonmises.natural_param,
            self._coef_gauss,
            self._coef_vm,
        )

    @property
    def dual_param(self) -> Tuple[Array, Array, float, float]:
        return self._gaussian.dual_param, self._vonmises.dual_param, self._coef_gauss, self._coef_vm

    def log_pdf(self, x: Array) -> Array:
        x = self._validate_input_samples(x)
        x_gauss, x_vm = self._split_input(x)

        alpha = self._coef_gauss
        beta = self._coef_vm
        log_2pi = np.log(2 * np.pi)

        # Gaussian
        mean, cov = self._gaussian.params
        d = self._gaussian.d
        chol = np.linalg.cholesky(cov)
        log_det = 2 * np.sum(np.log(np.diag(chol)))
        log_gauss_normalizer = 0.5 * (d * np.log(alpha)
                                      - d * (1 - alpha) * log_2pi
                                      - (1 - alpha) * log_det)
        # Von Mises
        _, kappa = self._vonmises.params
        log_vm_normalizer = (beta - 1) * log_2pi + beta * np.log(i0e(kappa)) + kappa \
                            - np.log(i0e(beta * kappa)) - beta * kappa

        log_pdf_gauss_alpha = alpha * self._gaussian.log_pdf(x_gauss) + log_gauss_normalizer
        log_pdf_vm_beta = beta * self._vonmises.log_pdf(x_vm) + log_vm_normalizer
        return log_pdf_gauss_alpha + log_pdf_vm_beta

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        self._validate_n_samples(n)
        raise NotImplementedError

    def fit(self,
            x: Array,
            sample_weight: Optional[Array] = None,
            case: str = "classic",
            ) -> "IndGaussVM":

        self._validate_case(case)
        x, sample_weight = self._input_process(x, sample_weight)
        x_gauss, x_vm = self._split_input(x)
        self._gaussian.fit(x_gauss, sample_weight=sample_weight, case=case)
        self._vonmises.fit(x_vm, sample_weight=sample_weight, case=case)

        return self

    @property
    def gaussian(self):
        return self._gaussian

    @property
    def vonmises(self):
        return self._vonmises
