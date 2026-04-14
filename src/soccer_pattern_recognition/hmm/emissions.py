"""Emission interfaces and implementations for modular HMMs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from .base_emission import BaseEmission
from ..core.types import Array

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

class GaussianEmission(BaseEmission):
    """
    Diagonal Gaussian emission model.

    Sufficient statistics:
    - ``post``: sum_t gamma_t(k)
    - ``obs``: sum_t gamma_t(k) * x_t
    - ``obs**2``: sum_t gamma_t(k) * x_t^2
    """

    param_symbols = "mc"

    def __init__(
        self,
        means: Array | None = None,
        covars: Array | None = None,
        *,
        min_covar: float = 1e-3,
        init_method: str = "kmeans",
    ) -> None:
        super().__init__()
        self.means_ = None if means is None else np.asarray(means, dtype=float)
        self.covars_ = None if covars is None else np.asarray(covars, dtype=float)
        self.min_covar = float(min_covar)
        self.init_method = str(init_method).lower()
        if self.min_covar <= 0:
            raise ValueError("min_covar must be strictly positive.")
        if self.init_method not in {"kmeans", "random"}:
            raise ValueError("init_method must be one of {'kmeans', 'random'}.")

    def get_n_fit_scalars_per_param(self) -> Mapping[str, int]:
        nc, nf = self._require_binding()
        return {"m": nc * nf, "c": nc * nf}

    def initialize(self, X: Array, init_params: str, random_state: Any) -> None:
        nc, nf = self._require_binding()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")
        if X.shape[1] != nf:
            raise ValueError(
                f"X must have {nf} features to match the bound emission, got {X.shape[1]}."
            )
        if X.shape[0] < 1:
            raise ValueError("X must contain at least one sample.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values.")
        random_state = check_random_state(random_state)

        needs_mean = "m" in init_params or self.means_ is None
        needs_covar = "c" in init_params or self.covars_ is None
        if not (needs_mean or needs_covar):
            return

        if self.init_method == "kmeans" and X.shape[0] >= nc:
            labels = KMeans(
                n_clusters=nc,
                random_state=random_state,
                n_init=10,
            ).fit_predict(X)
        else:
            labels = random_state.randint(nc, size=X.shape[0])

        global_var = np.var(X, axis=0) + self.min_covar
        if needs_mean:
            means = np.zeros((nc, nf), dtype=float)
            for k in range(nc):
                mask = labels == k
                if np.any(mask):
                    means[k] = X[mask].mean(axis=0)
                else:
                    means[k] = X[random_state.randint(X.shape[0])]
            self.means_ = means

        if needs_covar:
            covars = np.zeros((nc, nf), dtype=float)
            for k in range(nc):
                mask = labels == k
                if np.sum(mask) >= 2:
                    covars[k] = np.var(X[mask], axis=0) + self.min_covar
                else:
                    covars[k] = global_var
            self.covars_ = np.maximum(covars, self.min_covar)

    def check(self) -> None:
        nc, nf = self._require_binding()
        if self.means_ is None or self.covars_ is None:
            raise ValueError("GaussianEmission parameters are not initialized.")
        self.means_ = np.asarray(self.means_, dtype=float)
        self.covars_ = np.asarray(self.covars_, dtype=float)
        if self.means_.shape != (nc, nf):
            raise ValueError(f"means_ must have shape {(nc, nf)}.")
        if self.covars_.shape != (nc, nf):
            raise ValueError(f"covars_ must have shape {(nc, nf)}.")
        if not np.all(np.isfinite(self.means_)) or not np.all(np.isfinite(self.covars_)):
            raise ValueError("means_ and covars_ must contain only finite values.")
        if np.any(self.covars_ <= 0):
            raise ValueError("covars_ entries must be strictly positive.")
        if self.min_covar <= 0:
            raise ValueError("min_covar must be strictly positive.")

    def compute_log_likelihood(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        self.check()
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X must have {self.n_features} features, got {X.shape[1]}."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values.")
        diff = X[:, None, :] - self.means_[None, :, :]
        log_det = np.sum(np.log(2.0 * np.pi * self.covars_[None, :, :]), axis=2)
        quad = np.sum((diff * diff) / self.covars_[None, :, :], axis=2)
        return -0.5 * (log_det + quad)

    def initialize_sufficient_statistics(self) -> dict[str, Any]:
        nc, nf = self._require_binding()
        return {
            "post": np.zeros(nc, dtype=float),
            "obs": np.zeros((nc, nf), dtype=float),
            "obs**2": np.zeros((nc, nf), dtype=float),
        }

    def accumulate_sufficient_statistics(
        self,
        stats: dict[str, Any],
        X: Array,
        posteriors: Array,
        params: str,
    ) -> None:
        nc, nf = self._require_binding()
        X = np.asarray(X, dtype=float)
        posteriors = np.asarray(posteriors, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D when accumulating statistics.")
        if posteriors.ndim != 2:
            raise ValueError("posteriors must be 2D when accumulating statistics.")
        if X.shape[1] != nf:
            raise ValueError(f"X must have {nf} features, got {X.shape[1]}.")
        if posteriors.shape != (X.shape[0], nc):
            raise ValueError(
                "posteriors must have shape (n_samples, n_components), got "
                f"{posteriors.shape} for {(X.shape[0], nc)}."
            )

        if "m" in params or "c" in params:
            stats["post"] += posteriors.sum(axis=0)
        if "m" in params:
            stats["obs"] += posteriors.T @ X
        if "c" in params:
            stats["obs**2"] += posteriors.T @ (X ** 2)

    def m_step(self, stats: dict[str, Any], params: str) -> None:
        nc, nf = self._require_binding()
        if "m" not in params and "c" not in params:
            return
        if self.means_ is None:
            raise ValueError("means_ must be initialized before M-step.")
        if self.covars_ is None:
            raise ValueError("covars_ must be initialized before M-step.")

        post = np.asarray(stats["post"], dtype=float)
        obs = np.asarray(stats["obs"], dtype=float)
        obs2 = np.asarray(stats["obs**2"], dtype=float)
        if post.shape != (nc,):
            raise ValueError(f"stats['post'] must have shape {(nc,)}, got {post.shape}.")
        if obs.shape != (nc, nf):
            raise ValueError(f"stats['obs'] must have shape {(nc, nf)}, got {obs.shape}.")
        if obs2.shape != (nc, nf):
            raise ValueError(
                f"stats['obs**2'] must have shape {(nc, nf)}, got {obs2.shape}."
            )

        denom = np.maximum(post[:, None], 1e-12)
        active = post > 1e-12

        if "m" in params:
            candidate_means = obs / denom
            updated_means = np.asarray(self.means_, dtype=float).copy()
            updated_means[active] = candidate_means[active]
            self.means_ = updated_means

        if "c" in params:
            means = np.asarray(self.means_, dtype=float)
            second_moment = obs2 / denom
            candidate_covars = second_moment - means * means
            updated_covars = np.asarray(self.covars_, dtype=float).copy()
            updated_covars[active] = np.maximum(
                candidate_covars[active],
                self.min_covar,
            )
            self.covars_ = np.maximum(updated_covars, self.min_covar)

    def sample_from_state(self, state: int, random_state: Any) -> Array:
        nc, _ = self._require_binding()
        self.check()
        if state < 0 or state >= nc:
            raise ValueError(f"state must be in [0, {nc - 1}], got {state}.")
        random_state = check_random_state(random_state)
        return random_state.normal(self.means_[state], np.sqrt(self.covars_[state]))

