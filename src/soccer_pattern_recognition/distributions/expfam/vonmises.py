"""Von Mises exponential-family distribution on the unit circle."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.special import i0e, i1e
from scipy.optimize import minimize

from ..base import Array
from .base import ExponentialFamily


class VonMises(ExponentialFamily):
    """
    Von Mises distribution for directional data.
    x must be in sufficient statistic form: [cos x, sin x].
    """

    def __init__(self, loc: float = 0.0, kappa: float = 1.0):
        self._loc = float(loc)
        self._kappa = float(kappa)
        self._natural_param: Optional[Array] = None
        self._dual_param: Optional[Array] = None
        self._A: Optional[float] = None
        self._MAX_KAPPA = 50.0
        self._MAX_A = self._mean_length(self._MAX_KAPPA)  # A(50) = 0.9899489673784978
        self._validate()
        self._update_params()

    def _mean_length(self, kappa):
        """
        Mean resultant length define as i1(k)/i0(k) where i1 and i0 are the modified Bessel
        functions of first kind of order 1 and 0, respectively
        """
        kappa = np.clip(kappa, 1e-6, self._MAX_KAPPA)
        return i1e(kappa) / i0e(kappa)

    @staticmethod
    def _inv_mean_length(r: float):
        """
        A^{-1} approximation given by Best and Fisher (1981). Only for vonMises (d=2).
        """
        if r > 0.85:
            return 1.0 / (r ** 3 - 4 * r ** 2 + 3 * r)
        elif r > 0.53:
            return -0.4 + 1.39 * r + 0.43 / (1 - r)
        else:
            return 2 * r + r ** 3 + (5 / 6) * r ** 5

        # if r < 0.53:
        #     return 2 * r + r ** 3 + (5/6) * r ** 5
        # elif r < 0.85:
        #     return -0.4 + 1.39 * r + 0.43 / (1 - r)
        # else:
        #     return 1.0 / (r ** 3 - 4 * r ** 2 + 3 * r)

    @staticmethod
    def _inv_mean_length_v2(r: float):
        """
        A^{-1} approximation given by Banerjee (2005).
        """
        return r * (2 - r ** 2) / (1 - r ** 2)

    def _validate(self):
        if self._kappa <= 0:
            raise ValueError("Concentration parameter kappa must be positive.")

    def _update_params(self):
        self._natural_param = np.array([self._kappa * np.cos(self._loc),
                                        self._kappa * np.sin(self._loc)])
        self._A = self._mean_length(self._kappa)
        self._dual_param = np.array([self._A * np.cos(self._loc),
                                     self._A * np.sin(self._loc)])

    # ---- Getters and Setters ----
    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, value):
        self._loc = float(value)
        self._update_params()

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, value):
        self._kappa = value
        self._validate()
        self._update_params()

    @property
    def params(self) -> Tuple[float, float]:
        return self._loc, self._kappa

    @property
    def mean_length(self):
        return self._A

    @property
    def natural_param(self) -> Array:
        return self._natural_param.copy()

    @natural_param.setter
    def natural_param(self, theta: Array):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2,):
            raise ValueError("natural_param must be a length-2 vector.")
        if not np.all(np.isfinite(theta)):
            raise ValueError("natural_param contains non-finite values.")
        self._natural_param = theta
        self._loc = np.arctan2(theta[1], theta[0])
        self._kappa = np.minimum(np.linalg.norm(theta, ord=None), self._MAX_KAPPA)
        self._A = self._mean_length(self._kappa)
        self._validate()
        self._dual_param = np.array([self._A * np.cos(self._loc),
                                     self._A * np.sin(self._loc)])

    @property
    def dual_param(self) -> Array:
        return self._dual_param.copy()

    @dual_param.setter
    def dual_param(self, eta: Array):
        eta = np.asarray(eta, dtype=float)
        if eta.shape != (2,):
            raise ValueError("dual_param must be a length-2 vector.")
        if not np.all(np.isfinite(eta)):
            raise ValueError("dual_param contains non-finite values.")
        self._dual_param = eta
        self._loc = np.arctan2(eta[1], eta[0])
        self._A = np.minimum(np.linalg.norm(eta, ord=None), self._MAX_A)
        self._kappa = self._inv_mean_length(self._A)
        self._validate()
        self._natural_param = np.array([self._kappa * np.cos(self._loc),
                                        self._kappa * np.sin(self._loc)])

    @staticmethod
    def from_dual_to_ordinary(eta: Array) -> Tuple[Array, Array]:
        """
        Convert dual params eta = [eta1, eta2] to (loc, kappa) for one or many etas.
        If eta has shape (2,), returns array([loc, kappa]).
        If eta has shape (n,2), returns shape (n,2) with each row [loc, kappa].
        """
        eta = np.asarray(eta, float)
        single = (eta.ndim == 1)
        if single:
            eta = eta[np.newaxis, :]

        loc = np.arctan2(eta[:, 1], eta[:, 0])  # shape (n,)
        A = np.minimum(np.linalg.norm(eta, axis=1), 0.9899489673784978)  # A(50) = 0.9899489673784978

        # vectorized Best–Fisher inversion:
        #   if A<0.53: 2A + A^3 + 5A^5/6
        #   elif A<0.85: -0.4 + 1.39A + 0.43/(1−A)
        #   else: 1/(A^3 − 4A^2 + 3A)
        kappa = np.empty_like(A, dtype=float)

        m1 = A < 0.53
        m2 = (A >= 0.53) & (A < 0.85)
        m3 = ~(m1 | m2)

        a1 = A[m1]
        kappa[m1] = 2 * a1 + a1 ** 3 + (5 / 6) * a1 ** 5
        a2 = A[m2]
        kappa[m2] = -0.4 + 1.39 * a2 + 0.43 / (1 - a2)
        a3 = A[m3]
        kappa[m3] = 1.0 / (a3 ** 3 - 4 * a3 ** 2 + 3 * a3)

        return loc, kappa  # shape (n,) and (n,)

    # ----- densities -----
    def log_pdf(self, x: Array) -> Array:
        x = self._validate_input_samples(x)
        if x.ndim == 1:
            if x.shape[0] != 2:
                raise ValueError("VonMises expects x with shape (n, 2) or a single vector (2,).")
            x = x[np.newaxis, :]
        if x.shape[1] != 2:
            raise ValueError("VonMises expects x with shape (n, 2).")
        log_partition = np.log(2 * np.pi * i0e(self._kappa)) + self._kappa
        return x @ self._natural_param - log_partition

    # pdf inherited from base

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        """
        Draw samples and return sufficient statistics [cos(theta), sin(theta)].
        """
        self._validate_n_samples(n)
        rng = np.random.RandomState() if rng is None else rng
        theta = rng.vonmises(mu=self._loc, kappa=self._kappa, size=n)
        return np.column_stack((np.cos(theta), np.sin(theta)))

    # ----- Calibration -----
    def fit(
        self,
        x: Array,
        sample_weight: Optional[Array] = None,
        case: str = "bregman",
    ) -> "VonMises":

        self._validate_case(case)

        x, sample_weight = self._input_process(x, sample_weight)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("VonMises.fit expects x with shape (n, 2) in sufficient-stat form.")

        match case:
            case "bregman":
                # Compute dual/expectation parameters using sufficient statistics.
                eta = np.average(x, axis=0, weights=sample_weight)
                loc = np.arctan2(eta[1], eta[0])
                R = np.minimum(np.linalg.norm(eta, ord=None), self._MAX_A)
                kappa = self._inv_mean_length(R)

                self._loc, self._kappa = loc, kappa
                self._validate()
                self._update_params()

            case "classic":
                # Compute MLE with numerical optimizer
                const = np.log(2 * np.pi)

                def neg_ll(params):
                    loc, kappa = params
                    if kappa <= 0:
                        return np.inf
                    # i0e(kappa) = exp(-kappa)*i0(kappa)
                    # -log(i0(kappa)) = -log(i0e(kappa)) - kappa
                    ll = np.sum(sample_weight * (kappa * (np.cos(loc) * x[:, 0]
                                                          + np.sin(loc) * x[:, 1])
                                                 - np.log(i0e(kappa)) - kappa - const))
                    return -ll  # minimize negative

                C = np.sum(sample_weight * x[:, 0])
                S = np.sum(sample_weight * x[:, 1])
                initials = np.array([np.arctan2(S, C), self._kappa])
                bnds = ((-np.pi, np.pi), (1e-6, 50.0))
                result = minimize(
                    fun=neg_ll,
                    x0=initials,
                    method="L-BFGS-B",
                    bounds=bnds
                )
                self._loc, self._kappa = result.x
                self._validate()
                self._update_params()
        return self

    def __repr__(self):
        return f"VonMises(loc={self._loc * 180 / np.pi:.1f} deg, kappa={self._kappa:.3f})"
