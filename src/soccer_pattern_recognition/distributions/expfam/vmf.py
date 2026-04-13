"""Von Mises-Fisher exponential-family distribution on the unit sphere."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ive
from scipy.stats import vonmises_fisher

from ..base import Array
from .base import ExponentialFamily


class VonMisesFisher(ExponentialFamily):
    """
    Von Mises-Fisher distribution for unit vectors in R^d.

    Density:
        p(x | mu, kappa) = C_d(kappa) * exp(kappa * mu^T x),
    where ||mu|| = 1 and x lies on the unit hypersphere.
    """

    def __init__(self, d: int, *, mu: Optional[Array] = None, kappa: float = 1.0):
        super().__init__()
        if not isinstance(d, (int, np.integer)) or int(d) < 2:
            raise ValueError("d must be an integer >= 2.")
        self._d = int(d)
        if mu is None:
            self._mu = np.zeros(self._d, dtype=float)
            self._mu[0] = 1.0
        else:
            self._mu = np.asarray(mu, dtype=float)
        self._kappa = float(kappa)
        self._natural_param: Optional[Array] = None
        self._dual_param: Optional[Array] = None
        self._A: Optional[float] = None
        self._MIN_KAPPA = 1e-6
        self._MAX_KAPPA = 1e3
        self._validate()
        self._kappa = float(np.clip(self._kappa, self._MIN_KAPPA, self._MAX_KAPPA))
        self._MAX_A = float(self._mean_length(self._MAX_KAPPA))
        self._update_params()

    @property
    def d(self) -> int:
        return self._d

    @staticmethod
    def _mean_length_for_dimension(kappa: float, d: int, max_kappa: float = 1e3) -> float:
        """A_d(kappa) = I_{d/2}(kappa) / I_{d/2 - 1}(kappa)."""
        if d < 2:
            raise ValueError("dimension d must be >= 2 for VonMisesFisher.")
        if kappa <= 1e-6:
            return 0.0
        kappa = float(np.clip(kappa, 1e-6, max_kappa))
        nu_num = d / 2.0
        nu_den = d / 2.0 - 1.0
        vals = ive([nu_num, nu_den], kappa)
        return float(vals[0] / vals[1])

    def _mean_length(self, kappa: float) -> float:
        return self._mean_length_for_dimension(kappa, self._d, max_kappa=self._MAX_KAPPA)

    @staticmethod
    def _inv_mean_length(r: float, d: int, max_kappa: float = 1e3) -> float:
        """Numerical inverse of A_d using Banerjee init + Newton updates."""
        if not np.isfinite(r):
            raise ValueError("mean length r must be finite.")
        if r < 0.0:
            raise ValueError("mean length r must be nonnegative.")

        r = float(np.clip(r, 0.0, 1.0 - 1e-12))
        if r <= 1e-10:
            return 1e-6

        if d == 2:
            """
            A^{-1} approximation given by Best and Fisher (1981). Only for vonMises (d=2).
            """
            if r > 0.85:
                return 1.0 / (r ** 3 - 4 * r ** 2 + 3 * r)
            elif r > 0.53:
                return -0.4 + 1.39 * r + 0.43 / (1 - r)
            else:
                return 2 * r + r ** 3 + (5 / 6) * r ** 5

        denom = max(1.0 - r ** 2, 1e-12)
        #Banerjee initialization
        kappa = r * (d - r ** 2) / denom
        kappa = float(np.clip(kappa, 1e-6, max_kappa))

        for _ in range(30):
            A = VonMisesFisher._mean_length_for_dimension(kappa, d, max_kappa=max_kappa)
            dA = 1.0 - A ** 2 - ((d - 1.0) / kappa) * A
            if abs(dA) < 1e-12:
                break

            step = (A - r) / dA
            k_new = kappa - step
            if (not np.isfinite(k_new)) or k_new <= 0.0:
                k_new = 0.5 * kappa
            k_new = float(np.clip(k_new, 1e-6, max_kappa))

            if abs(k_new - kappa) <= 1e-8 * (1.0 + kappa):
                kappa = k_new
                break
            kappa = k_new

        return float(kappa)

    def _validate(self) -> None:
        if self._mu.ndim != 1 or self._mu.size != self._d:
            raise ValueError(f"mu must have shape ({self._d},).")
        if not np.all(np.isfinite(self._mu)):
            raise ValueError("mu contains non-finite values.")

        norm = float(np.linalg.norm(self._mu))
        if norm <= 0.0:
            raise ValueError("mu must be non-zero.")
        self._mu = self._mu / norm

        if not np.isfinite(self._kappa):
            raise ValueError("kappa must be finite.")
        if self._kappa <= 0.0:
            raise ValueError("Concentration parameter kappa must be positive.")

    @staticmethod
    def validate_unit_samples(x: Array, tol: float = 1e-6) -> None:
        norms = np.linalg.norm(x, axis=1)
        if not np.allclose(norms, 1.0, atol=tol, rtol=0.0):
            raise ValueError("All x samples must be unit vectors (||x|| = 1).")

    def _log_partition(self, kappa: float) -> float:
        """
        F(theta) for theta = kappa*mu:
            F = log((2*pi)^(d/2) * I_{d/2-1}(kappa) / kappa^(d/2-1))
        """
        kappa = float(np.clip(kappa, self._MIN_KAPPA, self._MAX_KAPPA))
        nu = 0.5 * self._d - 1.0
        ive_val = float(ive(nu, kappa))
        if ive_val <= np.finfo(float).tiny:
            ive_val = np.finfo(float).tiny
        log_bessel = np.log(ive_val) + kappa
        return 0.5 * self._d * np.log(2.0 * np.pi) + log_bessel - nu * np.log(kappa)

    def _update_params(self) -> None:
        self._kappa = float(np.clip(self._kappa, self._MIN_KAPPA, self._MAX_KAPPA))
        self._mu = self._mu / np.linalg.norm(self._mu)
        self._natural_param = self._kappa * self._mu
        self._A = self._mean_length(self._kappa)
        self._dual_param = self._A * self._mu

    # ---- Getters and Setters ----
    @property
    def mu(self) -> Array:
        return self._mu.copy()

    @mu.setter
    def mu(self, value: Array) -> None:
        self._mu = np.asarray(value, dtype=float)
        self._validate()
        self._update_params()

    @property
    def kappa(self) -> float:
        return float(self._kappa)

    @kappa.setter
    def kappa(self, value: float) -> None:
        self._kappa = float(value)
        self._validate()
        self._update_params()

    @property
    def params(self) -> Tuple[Array, float]:
        return self._mu.copy(), float(self._kappa)

    @params.setter
    def params(self, value: Tuple[Array, float]) -> None:
        mu, kappa = value
        self._mu = np.asarray(mu, dtype=float)
        self._kappa = float(kappa)
        self._validate()
        self._update_params()

    @property
    def mean_length(self) -> float:
        return float(self._A)

    @property
    def natural_param(self) -> Array:
        return self._natural_param.copy()

    @natural_param.setter
    def natural_param(self, theta: Array) -> None:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self._d,):
            raise ValueError(f"natural_param must have shape ({self._d},).")
        if not np.all(np.isfinite(theta)):
            raise ValueError("natural_param contains non-finite values.")

        kappa = float(np.linalg.norm(theta))
        if kappa <= 0.0:
            kappa = self._MIN_KAPPA
            mu = self._mu.copy()
        else:
            mu = theta / kappa

        self._mu = mu
        self._kappa = float(np.clip(kappa, self._MIN_KAPPA, self._MAX_KAPPA))
        self._validate()
        self._update_params()

    @property
    def dual_param(self) -> Array:
        return self._dual_param.copy()

    @dual_param.setter
    def dual_param(self, eta: Array) -> None:
        eta = np.asarray(eta, dtype=float)
        if eta.shape != (self._d,):
            raise ValueError(f"dual_param must have shape ({self._d},).")
        if not np.all(np.isfinite(eta)):
            raise ValueError("dual_param contains non-finite values.")

        r = float(np.linalg.norm(eta))
        if r <= 0.0:
            mu = self._mu.copy()
            kappa = self._MIN_KAPPA
        else:
            mu = eta / r
            r = min(r, self._MAX_A)
            kappa = self._inv_mean_length(r, self._d, max_kappa=self._MAX_KAPPA)

        self._mu = mu
        self._kappa = kappa
        self._validate()
        self._update_params()

    @staticmethod
    def from_dual_to_ordinary(eta: Array) -> Tuple[Array, Array]:
        """
        Convert dual params eta to ordinary (mu, kappa).

        If eta has shape (d,), returns (mu[d], kappa[scalar]).
        If eta has shape (n, d), returns (mu[n,d], kappa[n]).
        """
        eta = np.asarray(eta, dtype=float)
        single = eta.ndim == 1
        if single:
            eta = eta[np.newaxis, :]
        if eta.ndim != 2 or eta.shape[1] < 2:
            raise ValueError("eta must have shape (d,) or (n, d) with d >= 2.")
        if not np.all(np.isfinite(eta)):
            raise ValueError("eta contains non-finite values.")

        n, d = eta.shape
        raw_norm = np.linalg.norm(eta, axis=1)
        max_a = float(VonMisesFisher._mean_length_for_dimension(1e3, d, max_kappa=1e3))
        r = np.minimum(raw_norm, max_a)

        mu = np.zeros((n, d), dtype=float)
        mask = raw_norm > 0.0
        mu[mask] = eta[mask] / raw_norm[mask, None]
        mu[~mask, 0] = 1.0

        kappa = np.array(
            [VonMisesFisher._inv_mean_length(float(ri), d, max_kappa=1e3) for ri in r],
            dtype=float,
        )

        if single:
            return mu[0], kappa[0]
        return mu, kappa

    # ---- densities ----
    def log_pdf(self, x: Array) -> Array:
        x = self._validate_input_samples(x)
        if x.ndim == 1:
            if x.shape[0] != self._d:
                raise ValueError(f"VonMisesFisher expects x with shape ({self._d},) or (n, {self._d}).")
            x = x[np.newaxis, :]
        if x.shape[1] != self._d:
            raise ValueError(f"VonMisesFisher expects x with shape (n, {self._d}).")
        self.validate_unit_samples(x)
        return x @ self._natural_param - self._log_partition(self._kappa)

    # pdf inherited from base

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        self._validate_n_samples(n)
        rng = np.random.RandomState() if rng is None else rng
        return vonmises_fisher(mu=self._mu, kappa=self._kappa).rvs(size=n, random_state=rng)

    # ---- Calibration ----
    def fit(
        self,
        x: Array,
        sample_weight: Optional[Array] = None,
        case: str = "bregman",
    ) -> "VonMisesFisher":

        self._validate_case(case)
        x, sample_weight = self._input_process(x, sample_weight)
        if x.ndim != 2 or x.shape[1] != self._d:
            raise ValueError(f"VonMisesFisher.fit expects x with shape (n, {self._d}).")

        self.validate_unit_samples(x)
        eta = np.average(x, axis=0, weights=sample_weight)
        self.dual_param = eta

        return self

    def __repr__(self) -> str:
        mu_str = np.array2string(self._mu, precision=3, separator=" ", suppress_small=True)
        return f"VonMisesFisher(d={self._d}, mu={mu_str}, kappa={self._kappa:.3f})"
