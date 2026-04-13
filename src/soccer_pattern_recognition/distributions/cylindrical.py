"""Cylindrical distribution."""

from typing import Optional, Tuple

import numpy as np

from .base import Array, Distribution
from .expfam import MultivariateGaussian, VonMisesFisher


class Cylindrical(Distribution):
    """Cylindrical model with conditional Gaussian and directional vMF parts."""

    def __init__(self, d_gauss: int, d_vmf: int):
        if not isinstance(d_gauss, (int, np.integer)) or int(d_gauss) < 1:
            raise ValueError("d_gauss must be an integer >= 1.")
        if not isinstance(d_vmf, (int, np.integer)) or int(d_vmf) < 2:
            raise ValueError("d_vmf must be an integer >= 2.")

        self._d_gauss = int(d_gauss)
        self._d_vmf = int(d_vmf)

        self._mu_gauss = np.zeros(self._d_gauss, dtype=float)
        self._cross_cov = np.ones((self._d_gauss, self._d_vmf), dtype=float)
        self._cond_cov = np.eye(self._d_gauss, dtype=float)

        self._cond_gauss = MultivariateGaussian(
            self._d_gauss,
            mean=np.zeros(self._d_gauss, dtype=float),
            covariance=self._cond_cov.copy(),
        )
        self._vmf = VonMisesFisher(self._d_vmf)

        self._validate_params()
        self._cache()

    # ---- Validation helpers ----
    def _validate_params(self) -> None:
        if self._mu_gauss.ndim != 1 or self._mu_gauss.shape != (self._d_gauss,):
            raise ValueError(f"mu_gauss must have shape ({self._d_gauss},).")
        if self._cross_cov.ndim != 2 or self._cross_cov.shape != (self._d_gauss, self._d_vmf):
            raise ValueError(f"cross_cov must have shape ({self._d_gauss}, {self._d_vmf}).")
        if self._cond_cov.ndim != 2 or self._cond_cov.shape != (self._d_gauss, self._d_gauss):
            raise ValueError(f"cond_cov must have shape ({self._d_gauss}, {self._d_gauss}).")

        if not np.all(np.isfinite(self._mu_gauss)):
            raise ValueError("mu_gauss contains non-finite values.")
        if not np.all(np.isfinite(self._cross_cov)):
            raise ValueError("cross_cov contains non-finite values.")
        if not np.all(np.isfinite(self._cond_cov)):
            raise ValueError("cond_cov contains non-finite values.")
        if not np.allclose(self._cond_cov, self._cond_cov.T):
            raise ValueError("cond_cov must be symmetric.")

    def _cache(self) -> None:
        try:
            self._chol = np.linalg.cholesky(self._cond_cov)
        except np.linalg.LinAlgError as e:
            raise ValueError("cond_cov must be positive-definite.") from e
        self._log_det = 2 * np.sum(np.log(np.diag(self._chol)))

    def _validate_blocks(self, x_gauss: Array, x_vmf: Array) -> Tuple[Array, Array]:
        x_gauss = self._validate_input_samples(x_gauss)
        x_vmf = self._validate_input_samples(x_vmf)
        if x_gauss.ndim != 2 or x_gauss.shape[1] != self._d_gauss:
            raise ValueError(f"x_gauss must have shape (n, {self._d_gauss}).")
        if x_vmf.ndim != 2 or x_vmf.shape[1] != self._d_vmf:
            raise ValueError(f"x_vmf must have shape (n, {self._d_vmf}).")
        if x_gauss.shape[0] != x_vmf.shape[0]:
            raise ValueError("x_gauss and x_vmf must have the same number of samples.")
        return x_gauss, x_vmf

    @staticmethod
    def _normalize_weights(sample_weight: Optional[Array], n_obs: int) -> Array:
        if sample_weight is None:
            return np.full(n_obs, 1.0 / n_obs, dtype=float)
        w = np.asarray(sample_weight, dtype=float)
        if w.ndim != 1 or w.shape[0] != n_obs:
            raise ValueError(f"sample_weight must have shape ({n_obs},).")
        if not np.all(np.isfinite(w)):
            raise ValueError("sample_weight contains non-finite values.")
        if np.any(w < 0.0):
            raise ValueError("sample_weight must be nonnegative.")
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("sample_weight must sum to a positive value.")
        return w / total

    # ---- Properties and setters ----
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
    def mu_gauss(self) -> Array:
        return self._mu_gauss.copy()

    @mu_gauss.setter
    def mu_gauss(self, value: Array) -> None:
        self._mu_gauss = np.asarray(value, dtype=float)
        self._validate_params()

    @property
    def cross_cov(self) -> Array:
        return self._cross_cov.copy()

    @cross_cov.setter
    def cross_cov(self, value: Array) -> None:
        self._cross_cov = np.asarray(value, dtype=float)
        self._validate_params()

    @property
    def cond_cov(self) -> Array:
        return self._cond_cov.copy()

    @cond_cov.setter
    def cond_cov(self, value: Array) -> None:
        self._cond_cov = np.asarray(value, dtype=float)
        self._validate_params()
        self._cache()
        self._cond_gauss.params = (np.zeros(self._d_gauss, dtype=float), self._cond_cov.copy())

    @property
    def cond_gauss(self) -> MultivariateGaussian:
        return self._cond_gauss

    @property
    def vmf(self) -> VonMisesFisher:
        return self._vmf

    @property
    def params(self) -> Tuple[Array, Array, Array, Tuple[Array, float]]:
        return (
            self._mu_gauss.copy(),
            self._cross_cov.copy(),
            self._cond_cov.copy(),
            self._vmf.params,
        )

    @params.setter
    def params(self, value: Tuple[Array, Array, Array, Tuple[Array, float]]) -> None:
        mu_gauss, cross_cov, cond_cov, vmf_params = value
        self._mu_gauss = np.asarray(mu_gauss, dtype=float)
        self._cross_cov = np.asarray(cross_cov, dtype=float)
        self._cond_cov = np.asarray(cond_cov, dtype=float)
        self._vmf.params = vmf_params
        self._validate_params()
        self._cache()
        self._cond_gauss.params = (np.zeros(self._d_gauss, dtype=float), self._cond_cov.copy())

    # ---- Distribution API ----
    def log_pdf(self, x: Array) -> Array:
        x = self._validate_input_samples(x)
        if x.ndim != 2:
            raise ValueError("Cylindrical expects x with shape (n, d_gauss + d_vmf).")
        if x.shape[1] != self.d_total:
            raise ValueError(f"x must have {self.d_total} features.")

        x_gauss = x[:, :self._d_gauss]
        x_vmf = x[:, self._d_gauss:]
        self._vmf.validate_unit_samples(x_vmf)

        kappa = self._vmf.kappa
        mu_vmf = self._vmf.mu
        mean_shift = self._mu_gauss + kappa * (x_vmf - mu_vmf) @ self._cross_cov.T

        diff = x_gauss - mean_shift
        y = np.linalg.solve(self._chol, diff.T)
        quad = np.sum(y * y, axis=0)
        gauss_logpdf = -0.5 * (self._d_gauss * np.log(2 * np.pi) + self._log_det + quad)
        vmf_logpdf = self._vmf.log_pdf(x_vmf)
        return gauss_logpdf + vmf_logpdf

    def log_pdf_blocks(self, x_gauss: Array, x_vmf: Array) -> Array:
        x_gauss, x_vmf = self._validate_blocks(x_gauss, x_vmf)
        return self.log_pdf(np.concatenate((x_gauss, x_vmf), axis=1))

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        self._validate_n_samples(n)
        rng = np.random.RandomState() if rng is None else rng

        x_vmf = self._vmf.sample(n, rng)
        kappa = self._vmf.kappa
        mu_vmf = self._vmf.mu
        mean_shift = self._mu_gauss + kappa * (x_vmf - mu_vmf) @ self._cross_cov.T

        x_gauss = self._cond_gauss.sample(n, rng) + mean_shift
        return np.concatenate([x_gauss, x_vmf], axis=1)

    # ---- Calibration ----
    def fit(
        self,
        x_gauss: Array,
        x_vmf: Array,
        sample_weight: Optional[Array] = None,
        ridge: float = 1e-6,
    ) -> "Cylindrical":
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be a finite nonnegative scalar.")

        x_gauss, x_vmf = self._validate_blocks(x_gauss, x_vmf)
        n_obs = x_gauss.shape[0]
        weights = self._normalize_weights(sample_weight, n_obs)

        self._vmf.fit(x_vmf, sample_weight=weights, case="bregman")
        kappa = self._vmf.kappa
        mu_vmf = self._vmf.mu

        regressor = kappa * (x_vmf - mu_vmf)
        X = np.concatenate((np.ones((n_obs, 1), dtype=float), regressor), axis=1)

        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, None]
        Yw = x_gauss * sqrt_w[:, None]

        xtx = Xw.T @ Xw
        penalty = np.eye(xtx.shape[0], dtype=float)
        penalty[0, 0] = 0.0
        w = np.linalg.solve(xtx + ridge * penalty, Xw.T @ Yw)

        self._mu_gauss = w[0, :]
        self._cross_cov = w[1:, :].T

        residual = x_gauss - X @ w
        self._cond_cov = residual.T @ (weights[:, None] * residual)
        if ridge > 0.0:
            self._cond_cov += ridge * np.eye(self._d_gauss, dtype=float)

        self._validate_params()
        self._cache()
        self._cond_gauss.params = (np.zeros(self._d_gauss, dtype=float), self._cond_cov.copy())
        return self

    def __repr__(self) -> str:
        mu_gauss = np.array2string(self._mu_gauss, precision=4, separator=" ", suppress_small=True)
        cross_cov = np.array2string(self._cross_cov, precision=4, separator=" ", suppress_small=True)
        cond_cov = np.array2string(self._cond_cov, precision=4, separator=" ", suppress_small=True)
        vmf_mu = np.array2string(self._vmf.mu, precision=4, separator=" ", suppress_small=True)
        return (
            "Cylindrical(\n"
            f"  d_gauss={self._d_gauss}, d_vmf={self._d_vmf},\n"
            f"  mu_gauss={mu_gauss},\n"
            f"  cross_cov={cross_cov},\n"
            f"  cond_cov={cond_cov},\n"
            f"  vmf_mu={vmf_mu},\n"
            f"  vmf_kappa={self._vmf.kappa:.6g}\n"
            ")"
        )
