"""Cylindrical k-means for mixed Euclidean and spherical observations."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..core.types import Array, ArrayLike

RandomStateLike = Optional[Union[int, np.random.RandomState]]
_RESULTANT_DENOM_EPS = 1e-12


def _resolve_rng(rng: RandomStateLike) -> np.random.RandomState:
    if rng is None:
        return np.random.RandomState(42)
    if isinstance(rng, int):
        return np.random.RandomState(int(rng))
    if isinstance(rng, np.random.RandomState):
        return rng
    raise TypeError("rng must be None, an int seed, or np.random.RandomState.")


def _as_2d(x: ArrayLike, *, name: str) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    return arr


def _validate_inputs(
        x_euclid: ArrayLike,
        x_spherical: ArrayLike,
        *,
        n_clusters: int,
) -> tuple[Array, Array]:
    x1 = _as_2d(x_euclid, name="x_euclid")
    x2 = _as_2d(x_spherical, name="x_spherical")

    if x1.shape[0] < 1:
        raise ValueError("x_euclid must contain at least one sample.")
    if x1.shape[0] != x2.shape[0]:
        raise ValueError("x_euclid and x_spherical must have the same number of samples.")
    if x1.shape[0] < n_clusters:
        raise ValueError(
            f"n_clusters must be <= n_samples, got n_clusters={n_clusters}, n_samples={x1.shape[0]}."
        )
    if not np.all(np.isfinite(x1)):
        raise ValueError("x_euclid contains non-finite values.")
    if not np.all(np.isfinite(x2)):
        raise ValueError("x_spherical contains non-finite values.")

    norms = np.linalg.norm(x2, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("x_spherical contains at least one zero vector.")
    x2_unit = x2 / norms
    return x1, x2_unit


class CylindricalKMeans:
    """
    K-means for cylindrical data with Euclidean and spherical blocks.

    Assignment objective:
        (1/2d1) * (x1 - mu1_k)^T Sigma_1^{-1} (x1 - mu1_k)
        + lambda_ * (1 - x2^T mu2_k) / (1 - A_m^2)
    where x2 and mu2_k are unit vectors.
    Sigma_1 and A_m are estimated once from the full dataset per fit call.
    d1 is the Euclidean block dimension.
    """

    def __init__(
            self,
            n_clusters: int,
            *,
            lambda_: float = 1.0,
            max_iter: int = 100,
            tol: float = 1e-6,
            n_init: int = 10,
            rng: RandomStateLike = None,
    ) -> None:
        if not isinstance(n_clusters, (int, np.integer)) or int(n_clusters) < 1:
            raise ValueError("n_clusters must be an integer >= 1.")
        if not np.isfinite(lambda_) or float(lambda_) < 0.0:
            raise ValueError("lambda_ must be a finite nonnegative float.")
        if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
            raise ValueError("max_iter must be an integer >= 1.")
        if not isinstance(n_init, (int, np.integer)) or int(n_init) < 1:
            raise ValueError("n_init must be an integer >= 1.")
        if not np.isfinite(tol) or float(tol) < 0.0:
            raise ValueError("tol must be a finite nonnegative float.")

        self.n_clusters = int(n_clusters)
        self.lambda_ = float(lambda_)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self._rng = _resolve_rng(rng)

        self.cluster_centers_euclid_: Optional[Array] = None
        self.cluster_centers_spherical_: Optional[Array] = None
        self.labels_: Optional[Array] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.linear_covariance_: Optional[Array] = None
        self.linear_precision_: Optional[Array] = None
        self.linear_dimension_: Optional[int] = None
        self.spherical_resultant_length_: Optional[float] = None

    @staticmethod
    def _linear_covariance_and_precision(x1: Array) -> tuple[Array, Array]:
        centered = x1 - np.mean(x1, axis=0, keepdims=True)
        denom = float(max(x1.shape[0], 1))
        covariance = (centered.T @ centered) / denom
        # Use pseudo-inverse so the method remains well-defined for singular covariance.
        precision = np.linalg.pinv(covariance)
        precision = 0.5 * (precision + precision.T)
        return covariance, precision

    @staticmethod
    def _mean_resultant_length(x2: Array) -> float:
        if x2.shape[0] == 0:
            return 0.0
        return min(float(np.linalg.norm(np.mean(x2, axis=0), ord=2)), 1.0 - 1e-6)

    @staticmethod
    def _euclidean_mahalanobis(
            diff: Array,
            precision: Array,
            *,
            scale: float,
    ) -> Array:
        return scale * np.sum((diff @ precision) * diff, axis=-1)

    @staticmethod
    def _directional_mahalanobis(
            inner_product: Array,
            resultant_length: float,
    ) -> Array:
        cos_sim = np.clip(inner_product, -1.0, 1.0)
        denom = max(1.0 - resultant_length * resultant_length, _RESULTANT_DENOM_EPS)
        return (1.0 - cos_sim) / denom

    def _init_centers(
            self,
            x1: Array,
            x2: Array,
    ) -> tuple[Array, Array]:
        idx = self._rng.choice(x1.shape[0], self.n_clusters, replace=False)
        return x1[idx].copy(), x2[idx].copy()

    def _assign(
            self,
            x1: Array,
            x2: Array,
            mu1: Array,
            mu2: Array,
            sigma1_inv: Array,
            resultant_length: float,
            linear_scale: float,
    ) -> tuple[Array, Array]:
        diff = x1[:, None, :] - mu1[None, :, :]
        linear_term = self._euclidean_mahalanobis(diff, sigma1_inv, scale=linear_scale)
        dir_term = self._directional_mahalanobis(x2 @ mu2.T, resultant_length)
        loss = linear_term + self.lambda_ * dir_term
        labels = np.argmin(loss, axis=1)
        point_loss = loss[np.arange(x1.shape[0]), labels]
        return labels, point_loss

    def _update_centers(
            self,
            x1: Array,
            x2: Array,
            labels: Array,
            point_loss: Array,
    ) -> tuple[Array, Array, bool]:
        mu1 = np.zeros((self.n_clusters, x1.shape[1]), dtype=float)
        mu2 = np.zeros((self.n_clusters, x2.shape[1]), dtype=float)
        had_empty_reseed = False

        # Re-seed empty clusters with the worst represented observations.
        farthest_order = np.argsort(-point_loss)
        farthest_ptr = 0

        for k in range(self.n_clusters):
            mask = labels == k
            if not np.any(mask):
                had_empty_reseed = True
                idx = int(farthest_order[farthest_ptr])
                farthest_ptr += 1
                mu1[k] = x1[idx]
                mu2[k] = x2[idx]
                continue

            mu1[k] = np.mean(x1[mask], axis=0)
            resultant = np.sum(x2[mask], axis=0)
            norm = float(np.linalg.norm(resultant))
            if norm <= 0.0:
                idx = int(self._rng.choice(x1.shape[0]))
                mu2[k] = x2[idx]
            else:
                mu2[k] = resultant / norm

        return mu1, mu2, had_empty_reseed

    def fit(self, x_euclid: ArrayLike, x_spherical: ArrayLike) -> "CylindricalKMeans":
        x1, x2 = _validate_inputs(
            x_euclid,
            x_spherical,
            n_clusters=self.n_clusters,
        )
        sigma1, sigma1_inv = self._linear_covariance_and_precision(x1)
        d1 = float(x1.shape[1])
        linear_scale = 0.5 / d1

        resultant_length = self._mean_resultant_length(x2)

        best_mu1: Optional[Array] = None
        best_mu2: Optional[Array] = None
        best_labels: Optional[Array] = None
        best_inertia = np.inf
        best_n_iter = 0

        for _ in range(self.n_init):
            mu1, mu2 = self._init_centers(x1, x2)
            labels = np.full(x1.shape[0], -1, dtype=int)
            n_iter_run = 0

            for it in range(self.max_iter):
                new_labels, point_loss = self._assign(
                    x1,
                    x2,
                    mu1,
                    mu2,
                    sigma1_inv,
                    resultant_length,
                    linear_scale,
                )
                new_mu1, new_mu2, had_empty_reseed = self._update_centers(
                    x1,
                    x2,
                    new_labels,
                    point_loss,
                )

                linear_shift = self._euclidean_mahalanobis(
                    mu1 - new_mu1,
                    sigma1_inv,
                    scale=linear_scale,
                )
                directional_shift = self._directional_mahalanobis(
                    np.sum(mu2 * new_mu2, axis=1),
                    resultant_length,
                )

                shift = float(np.max(linear_shift + self.lambda_ * directional_shift))

                mu1 = new_mu1
                mu2 = new_mu2
                n_iter_run = it + 1
                if (not had_empty_reseed) and (
                        np.array_equal(new_labels, labels) or shift <= self.tol
                ):
                    labels = new_labels
                    break
                labels = new_labels

            final_labels, final_point_loss = self._assign(
                x1,
                x2,
                mu1,
                mu2,
                sigma1_inv,
                resultant_length,
                linear_scale,
            )
            final_inertia = float(np.sum(final_point_loss))

            if final_inertia < best_inertia:
                best_inertia = final_inertia
                best_mu1 = mu1.copy()
                best_mu2 = mu2.copy()
                best_labels = final_labels.copy()
                best_n_iter = n_iter_run

        if best_mu1 is None or best_mu2 is None or best_labels is None:
            raise RuntimeError("Cylindrical k-means failed to find a valid clustering.")

        self.cluster_centers_euclid_ = best_mu1
        self.cluster_centers_spherical_ = best_mu2
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = int(best_n_iter)
        self.linear_covariance_ = sigma1
        self.linear_precision_ = sigma1_inv
        self.linear_dimension_ = int(d1)
        self.spherical_resultant_length_ = resultant_length
        return self

    def predict(self, x_euclid: ArrayLike, x_spherical: ArrayLike) -> Array:
        if (
                self.cluster_centers_euclid_ is None
                or self.cluster_centers_spherical_ is None
                or self.linear_precision_ is None
                or self.linear_dimension_ is None
                or self.spherical_resultant_length_ is None
        ):
            raise RuntimeError("The model is not fitted yet. Call fit(...) first.")

        x1, x2 = _validate_inputs(x_euclid, x_spherical, n_clusters=1)
        if x1.shape[1] != self.cluster_centers_euclid_.shape[1]:
            raise ValueError(
                "x_euclid feature mismatch with fitted centers: "
                f"expected {self.cluster_centers_euclid_.shape[1]}, got {x1.shape[1]}."
            )
        if x2.shape[1] != self.cluster_centers_spherical_.shape[1]:
            raise ValueError(
                "x_spherical feature mismatch with fitted centers: "
                f"expected {self.cluster_centers_spherical_.shape[1]}, got {x2.shape[1]}."
            )
        linear_scale = 0.5 / float(self.linear_dimension_)

        labels, _ = self._assign(
            x1,
            x2,
            self.cluster_centers_euclid_,
            self.cluster_centers_spherical_,
            self.linear_precision_,
            self.spherical_resultant_length_,
            linear_scale,
        )
        return labels

    def fit_predict(self, x_euclid: ArrayLike, x_spherical: ArrayLike) -> Array:
        self.fit(x_euclid, x_spherical)
        if self.labels_ is None:
            raise RuntimeError("fit(...) finished without labels.")
        return self.labels_.copy()


def cylindrical_kmeans(
        x_euclid: ArrayLike,
        x_spherical: ArrayLike,
        n_clusters: int,
        *,
        lambda_: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_init: int = 10,
        rng: RandomStateLike = None,
) -> tuple[Array, Array, Array, float]:
    """
    Functional wrapper for cylindrical k-means.

    Returns
    -------
    labels : Array, shape (n_samples,)
        Hard cluster assignments.
    centers_euclid : Array, shape (n_clusters, d1)
        Euclidean centroids.
    centers_spherical : Array, shape (n_clusters, d2)
        Unit-norm spherical centroids.
    inertia : float
        Sum of assignment losses.
    """
    model = CylindricalKMeans(
        n_clusters=n_clusters,
        lambda_=lambda_,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        rng=rng,
    )
    model.fit(x_euclid, x_spherical)
    if (
            model.cluster_centers_euclid_ is None
            or model.cluster_centers_spherical_ is None
            or model.labels_ is None
            or model.inertia_ is None
    ):
        raise RuntimeError("Cylindrical k-means did not produce a complete result.")

    return (
        model.labels_.copy(),
        model.cluster_centers_euclid_.copy(),
        model.cluster_centers_spherical_.copy(),
        float(model.inertia_),
    )


__all__ = ["CylindricalKMeans", "cylindrical_kmeans"]
