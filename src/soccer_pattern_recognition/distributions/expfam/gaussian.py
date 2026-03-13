"""Univariate and multivariate Gaussian exponential-family distributions."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...core import _EPS
from ..base import Array
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
            if x.shape[1] != 1:
                raise ValueError("UnivariateGaussian expects x with shape (n,) or (n, 1).")
            x = x[:, 0]
        return -0.5 * ((x - self._mean) ** 2) / self._variance - 0.5 * np.log(2 * np.pi * self._variance)

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        self._validate_n_samples(n)
        rng = np.random.RandomState() if rng is None else rng
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
            if x.shape[1] != 1:
                raise ValueError("UnivariateGaussian.fit expects x with shape (n,) or (n, 1).")
            x = x[:, 0]
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


# -------------------- Multivariate Gaussian --------------------
class MultivariateGaussian(ExponentialFamily):
    """
    Multivariate Gaussian N(mu, sigma), with mu = mean vector and sigma = covariance matrix
    """

    def __init__(self, mean: Optional[Array] = None,
                 covariance: Optional[Array] = None):
        super().__init__()
        self._mean = np.zeros(2) if mean is None else np.asarray(mean, dtype=float)
        self._covariance = np.eye(self._mean.size) if covariance is None else np.asarray(covariance, dtype=float)
        self._validate()
        self._cache()

    def _validate(self):
        if self._covariance.shape[0] != self._covariance.shape[1]:
            raise ValueError("Covariance matrix must be square.")
        if self._mean.shape[0] != self._covariance.shape[0]:
            raise ValueError("Mean and covariance shapes mismatch.")
        if not np.allclose(self._covariance, self._covariance.T):
            raise ValueError("Covariance must be symmetric.")
        # Positive definiteness checked in _cache via Cholesky

    def _cache(self):
        try:
            self._chol = np.linalg.cholesky(self._covariance)
        except np.linalg.LinAlgError as e:
            raise ValueError("Covariance must be positive-definite.") from e
        self._log_det = 2 * np.sum(np.log(np.diag(self._chol)))

    # ---- Getters and Setter ----
    @property
    def d(self) -> int:
        return self._mean.size

    @property
    def params(self) -> Tuple[Array, Array]:
        return self._mean.copy(), self._covariance.copy()

    @params.setter
    def params(self, value: Tuple[Array, Array]):
        mean, covariance = value
        self._mean = np.asarray(mean, dtype=float)
        self._covariance = np.asarray(covariance, dtype=float)
        self._validate()
        self._cache()

    @property
    def natural_param(self):
        theta_1 = np.linalg.solve(self._covariance, self._mean)
        theta_2 = -0.5 * np.linalg.inv(self._covariance)
        return np.concatenate([theta_1, theta_2.ravel()])

    @natural_param.setter
    def natural_param(self, theta: Array):
        raise NotImplementedError("Setting natural_param is not implemented.")

    @property
    def dual_param(self) -> Array:
        mu = self._mean
        second_moment = (self._covariance + np.outer(mu, mu)).flatten()
        return np.concatenate([mu, second_moment.ravel()])

    @dual_param.setter
    def dual_param(self, eta: Array):
        eta = np.asarray(eta, dtype=float)
        d = self.d
        if eta.shape != (d + d * d,):
            raise ValueError(f"dual_param must have shape ({d + d * d},).")
        mu = eta[:d]  # E[x]
        second_moments = eta[d:].reshape((d, d))  # E[x x^T]
        cov = second_moments - np.outer(mu, mu)  # covariance = E[x x^T] – mu mu^T
        cov = 0.5 * (cov + cov.swapaxes(-1, -2))  # ensure symmetric matrix
        cov += _EPS * np.eye(cov.shape[0])  # Numerical jitter if near-singular
        self._mean, self._covariance = mu, cov
        self._validate()
        self._cache()

    @staticmethod
    def get_sufficient_stat(x: Array) -> Array:
        """
        Get the sufficient statistic vector e.g. case d=2: [x y x^2 xy yx y^2]
        :return: array of shape (N,d+d^2)
        """
        n = x.shape[0]
        d = x.shape[1]
        outer = np.einsum('ij,ik->ijk', x, x)  # (n,d,d)
        return np.concatenate([x, outer.reshape(n, d ** 2)], axis=1)

    @staticmethod
    def from_dual_to_ordinary(eta: Array) -> Tuple[Array, Array]:
        """
        Vectorized conversion from dual parameters to (mean, covariance).
        Returns:
            mu: array of shape (n, d)
            cov: array of shape (n, d, d)
        """
        eta = np.asarray(eta, dtype=float)
        if eta.ndim == 1:
            eta = eta[None, :]  # promote single vector to batch

        n, L = eta.shape
        # solve d from L = d + d^2  => d = (-1 + sqrt(1+4L)) / 2
        d = int(-0.5 + 0.5 * np.sqrt(1 + 4 * L))
        if d + d * d != L:
            raise ValueError(f"Invalid eta length {L}; cannot infer integer d.")

        mu = eta[:, :d]  # (n, d)
        second_moments = eta[:, d:].reshape(n, d, d)  # (n, d, d)
        # covariance = E[xx^T] - mu mu^T, broadcasting outer product
        cov = second_moments - mu[:, :, None] * mu[:, None, :]  # (n, d, d)
        cov = 0.5 * (cov + cov.swapaxes(-1, -2))  # ensure symmetric matrix
        cov = cov + _EPS * np.eye(d)

        return mu, cov

    # ----- densities -----
    def log_pdf(self, x: Array) -> Array:
        x = self._validate_input_samples(x)
        if x.ndim != 2:
            raise ValueError("MultivariateGaussian expects x with shape (n, d).")
        if x.shape[1] != self.d:
            raise ValueError(f"x has {x.shape[1]} features, expected {self.d}.")
        diff = x - self._mean  # (d,) or (n,d)
        # Solve L y = diff^T => y = L^{-1} diff^T
        y = np.linalg.solve(self._chol, diff.T)  # (d,N)
        quad = np.sum(y * y, axis=0)  # (N,)
        return -0.5 * (self.d * np.log(2 * np.pi) + self._log_det + quad)

    # pdf inherited from base

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        self._validate_n_samples(n)
        rng = np.random.RandomState() if rng is None else rng
        return rng.multivariate_normal(self._mean, self._covariance, size=n)

    # ----- Calibration -----
    def fit(self,
            x: Array,
            sample_weight: Optional[Array] = None,
            case: str = "classic",
            ) -> "MultivariateGaussian":
        self._validate_case(case)
        x, sample_weight = self._input_process(x, sample_weight)
        if x.ndim != 2:
            raise ValueError("MultivariateGaussian.fit expects x with shape (n, d).")
        if x.shape[1] != self.d:
            raise ValueError(f"x has {x.shape[1]} features, expected {self.d}.")
        match case:
            case "bregman":
                # Compute MLE via minimization of Bregman divergence
                # form sufficient stats and average
                suf_stat = self.get_sufficient_stat(x)  # shape (n, d + d^2)
                dual = np.average(suf_stat, axis=0, weights=sample_weight)  # length d + d^2
                # update params
                self.dual_param = dual
                self._validate()
                self._cache()
            case _:
                # Compute MLE via analytical solution of ordinary-coordinates parameters
                mu = np.average(x, axis=0, weights=sample_weight)
                diff = x - mu
                # Broadcasting weights to columns; (n,1) * (n,d) -> weighted rows
                weighted_diff = sample_weight[:, np.newaxis] * diff
                cov = weighted_diff.T @ diff
                cov += _EPS * np.eye(cov.shape[0])  # Numerical jitter if near-singular

                self._mean = mu
                self._covariance = cov
                self._validate()
                self._cache()
        return self

    def __repr__(self):
        mean_str = np.array2string(self._mean, precision=3, separator=' ', suppress_small=True)
        cov_rows = [np.array2string(row, precision=3, separator=' ', suppress_small=True) for row in self._covariance]
        cov_str = '[' + ', '.join(cov_rows) + ']'
        return f"MultivariateGaussian(d={self.d}, mean={mean_str}, cov={cov_str})"
