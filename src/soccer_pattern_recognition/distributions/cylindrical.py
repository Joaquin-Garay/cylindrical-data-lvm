"""Cylindrical distribution."""

from typing import Optional, Tuple

import numpy as np

from ..core.types import Array
from .base import Distribution
from .expfam import MultivariateGaussian, VonMisesFisher


class Cylindrical(Distribution):
    """Cylindrical model with conditional Gaussian and directional vMF parts."""

    def __init__(self, d_gauss: int,
                 d_vmf: int,
                 *,
                 mu_gauss: Optional[Array] = None,
                 cross_cov: Optional[Array] = None,
                 cond_cov: Optional[Array] = None,
                 mu_vmf: Optional[Array] = None,
                 kappa: Optional[float] = None):

        self._d_gauss = self._validate_positive_int(d_gauss, name="d_gauss", minimum=1)
        self._d_vmf = self._validate_positive_int(d_vmf, name="d_vmf", minimum=2)

        self._mu_gauss = (
            np.zeros(self._d_gauss, dtype=float)
            if mu_gauss is None
            else np.asarray(mu_gauss, dtype=float)
        )
        self._cross_cov = (
            np.ones((self._d_gauss, self._d_vmf), dtype=float)
            if cross_cov is None
            else np.asarray(cross_cov, dtype=float)
        )
        self._cond_cov = (
            np.eye(self._d_gauss, dtype=float)
            if cond_cov is None
            else np.asarray(cond_cov, dtype=float)
        )

        vmf_mu = None if mu_vmf is None else np.asarray(mu_vmf, dtype=float)
        vmf_kappa = 1.0 if kappa is None else float(kappa)
        self._vmf = VonMisesFisher(self._d_vmf, mu=vmf_mu, kappa=vmf_kappa)

        self._cond_gauss = MultivariateGaussian(
            self._d_gauss,
            mean=np.zeros(self._d_gauss, dtype=float),
            covariance=self._cond_cov.copy(),
        )

        self._validate_params()
        self._cache()

    # ---- Validation helpers ----
    def _validate_params(self) -> None:
        self._mu_gauss = self._validate_vector(
            self._mu_gauss,
            size=self._d_gauss,
            name="mu_gauss",
        )
        self._cross_cov = self._validate_matrix(
            self._cross_cov,
            shape=(self._d_gauss, self._d_vmf),
            name="cross_cov",
        )
        self._cond_cov = self._validate_matrix(
            self._cond_cov,
            shape=(self._d_gauss, self._d_gauss),
            name="cond_cov",
            symmetric=True,
        )

    def _cache(self) -> None:
        try:
            self._chol = np.linalg.cholesky(self._cond_cov)
        except np.linalg.LinAlgError as e:
            raise ValueError("cond_cov must be positive-definite.") from e
        self._log_det = 2 * np.sum(np.log(np.diag(self._chol)))

    def _validate_blocks(self, x_gauss: Array, x_vmf: Array) -> Tuple[Array, Array]:
        x_gauss = self._validate_input_matrix(x_gauss, n_features=self._d_gauss, name="x_gauss")
        x_vmf = self._validate_input_matrix(x_vmf, n_features=self._d_vmf, name="x_vmf")
        if x_gauss.shape[0] != x_vmf.shape[0]:
            raise ValueError("x_gauss and x_vmf must have the same number of samples.")
        return x_gauss, x_vmf

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
        x = self._validate_input_matrix(
            x,
            n_features=self.d_total,
            name="x",
        )

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
        rng = self._resolve_rng(rng)

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
        x_vmf: Optional[Array] = None,
        sample_weight: Optional[Array] = None,
        case: str = None,
        ridge: float = 1e-6,
    ) -> "Cylindrical":
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be a finite nonnegative scalar.")
        if x_vmf is None:
            x_joint = self._validate_input_matrix(
                x_gauss,
                n_features=self.d_total,
                name="x",
            )
            x_gauss, x_vmf = x_joint[:, :self._d_gauss], x_joint[:, self._d_gauss:]

        x_gauss, x_vmf = self._validate_blocks(x_gauss, x_vmf)
        n_obs = x_gauss.shape[0]
        weights = self._normalize_sample_weight(sample_weight, n_obs)

        self._vmf.fit(x_vmf, sample_weight=weights, case="bregman")
        kappa = self._vmf.kappa
        mu_vmf = self._vmf.mu

        regressor = kappa * (x_vmf - mu_vmf)
        X = np.concatenate((np.ones((n_obs, 1), dtype=float), regressor), axis=1)

        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, None]
        Yw = x_gauss * sqrt_w[:, None]

        ridge *= kappa**2 # this keeps regularization invariant of kappa values.
        xtx = Xw.T @ Xw
        penalty = np.eye(xtx.shape[0], dtype=float)
        penalty[0, 0] = 0.0
        w = np.linalg.solve(xtx + ridge * penalty, Xw.T @ Yw)

        self._mu_gauss = w[0, :]
        self._cross_cov = w[1:, :].T

        residual = x_gauss - X @ w
        self._cond_cov = residual.T @ (weights[:, None] * residual)
        if ridge > 0.0:
            self._cond_cov += (ridge/(kappa**2)) * np.eye(self._d_gauss, dtype=float)

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
            f"Cylindrical({self._d_gauss}, {self._d_vmf}):\n"
            f" mu_gauss={mu_gauss}\n"
            f" cross_cov={cross_cov}\n"
            f" cond_cov={cond_cov}\n"
            f" vmf_mu={vmf_mu}\n"
            f" vmf_kappa={self._vmf.kappa:.6g}"
        )
