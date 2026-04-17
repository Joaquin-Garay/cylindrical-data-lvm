"""Spherical k-means clustering with cosine-similarity assignments."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..core.types import Array


RandomStateLike = Optional[Union[int, np.random.RandomState]]


def _resolve_rng(rng: RandomStateLike) -> np.random.RandomState:
    if rng is None:
        return np.random.RandomState(42)
    if isinstance(rng, int):
        return np.random.RandomState(int(rng))
    if isinstance(rng, np.random.RandomState):
        return rng
    raise TypeError("rng must be None, an int seed, or np.random.RandomState.")


def _validate_input(x: Array, *, n_clusters: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("x must be a 2D array with shape (n_samples, n_features).")
    if arr.shape[0] < 1:
        raise ValueError("x must contain at least one sample.")
    if arr.shape[0] < n_clusters:
        raise ValueError(
            f"n_clusters must be <= n_samples, got n_clusters={n_clusters}, n_samples={arr.shape[0]}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains non-finite values.")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("x contains at least one zero vector; spherical k-means needs nonzero samples.")
    return arr / norms


class SphericalKMeans:
    """
    Spherical k-means using cosine-similarity assignments.

    At each iteration:
    - E-step: assign each sample to the center with largest dot product.
    - M-step: update each center as normalized sum of assigned samples.
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_init: int = 10,
        rng: RandomStateLike = None,
    ) -> None:
        if not isinstance(n_clusters, (int, np.integer)) or int(n_clusters) < 1:
            raise ValueError("n_clusters must be an integer >= 1.")
        if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
            raise ValueError("max_iter must be an integer >= 1.")
        if not isinstance(n_init, (int, np.integer)) or int(n_init) < 1:
            raise ValueError("n_init must be an integer >= 1.")
        if not np.isfinite(tol) or float(tol) < 0.0:
            raise ValueError("tol must be a finite nonnegative float.")

        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self._rng = _resolve_rng(rng)

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.score_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    def _init_centers(self, x: np.ndarray) -> np.ndarray:
        idx = self._rng.choice(x.shape[0], self.n_clusters, replace=False)
        return x[idx].copy()

    def _update_centers(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        sims: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        centers = np.zeros((self.n_clusters, x.shape[1]), dtype=float)
        had_empty_reseed = False
        # Re-seed empty clusters with least well represented points.
        farthest_order = np.argsort(np.max(sims, axis=1))
        farthest_ptr = 0
        used_fallback_idx: set[int] = set()

        for k in range(self.n_clusters):
            mask = labels == k
            if not np.any(mask):
                had_empty_reseed = True
                while (
                    farthest_ptr < x.shape[0]
                    and int(farthest_order[farthest_ptr]) in used_fallback_idx
                ):
                    farthest_ptr += 1
                if farthest_ptr < x.shape[0]:
                    idx = int(farthest_order[farthest_ptr])
                    farthest_ptr += 1
                else:
                    idx = int(self._rng.choice(x.shape[0]))
                centers[k] = x[idx]
                used_fallback_idx.add(idx)
                continue

            mean_dir = x[mask].sum(axis=0)
            norm = float(np.linalg.norm(mean_dir))
            if norm <= 0.0:
                idx = int(self._rng.choice(x.shape[0]))
                centers[k] = x[idx]
            else:
                centers[k] = mean_dir / norm

        return centers, had_empty_reseed

    def fit(self, x: Array) -> "SphericalKMeans":
        x_unit = _validate_input(x, n_clusters=self.n_clusters)

        best_centers: Optional[np.ndarray] = None
        best_labels: Optional[np.ndarray] = None
        best_score = -np.inf
        best_n_iter = 0

        for _ in range(self.n_init):
            centers = self._init_centers(x_unit)
            labels = np.full(x_unit.shape[0], -1, dtype=int)
            n_iter_run = 0

            for it in range(self.max_iter):
                sims = x_unit @ centers.T
                new_labels = np.argmax(sims, axis=1)
                new_centers, had_empty_reseed = self._update_centers(x_unit, new_labels, sims)

                center_cos = np.sum(centers * new_centers, axis=1)
                center_shift = float(np.max(1.0 - np.clip(center_cos, -1.0, 1.0)))

                centers = new_centers
                n_iter_run = it + 1
                if (not had_empty_reseed) and (
                    np.array_equal(new_labels, labels) or center_shift <= self.tol
                ):
                    labels = new_labels
                    break
                labels = new_labels

            final_sims = x_unit @ centers.T
            final_labels = np.argmax(final_sims, axis=1)
            final_score = float(np.sum(np.max(final_sims, axis=1)))

            if final_score > best_score:
                best_score = final_score
                best_centers = centers.copy()
                best_labels = final_labels.copy()
                best_n_iter = n_iter_run

        if best_centers is None or best_labels is None:
            raise RuntimeError("Spherical k-means failed to find a valid clustering.")

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.score_ = float(best_score)
        self.n_iter_ = int(best_n_iter)
        return self

    def predict(self, x: Array) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("The model is not fitted yet. Call fit(...) first.")
        x_unit = _validate_input(x, n_clusters=1)
        sims = x_unit @ self.cluster_centers_.T
        return np.argmax(sims, axis=1)

    def fit_predict(self, x: Array) -> np.ndarray:
        self.fit(x)
        if self.labels_ is None:
            raise RuntimeError("fit(...) finished without labels.")
        return self.labels_.copy()


def spherical_kmeans(
    x: Array,
    n_clusters: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    n_init: int = 10,
    rng: RandomStateLike = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Functional wrapper for spherical k-means.

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        Hard cluster assignments.
    centers : np.ndarray, shape (n_clusters, n_features)
        Unit-norm cluster centers.
    score : float
        Sum of maximum cosine similarities over samples.
    """
    model = SphericalKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        rng=rng,
    )
    model.fit(x)
    if model.cluster_centers_ is None or model.labels_ is None or model.score_ is None:
        raise RuntimeError("Spherical k-means did not produce a complete result.")
    return model.labels_.copy(), model.cluster_centers_.copy(), float(model.score_)


__all__ = ["SphericalKMeans", "spherical_kmeans"]
