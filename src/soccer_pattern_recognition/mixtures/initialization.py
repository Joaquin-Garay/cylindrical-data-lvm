"""Initialization methods for MixtureModel objects."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus

from ..core.types import Array

if TYPE_CHECKING:
    from ..mixtures.mixture import MixtureModel

def build_initial_posteriors(model: "MixtureModel",
                              x: Array,
                              sample_weight: Array,
                              rng: np.random.RandomState,
                              n_samples: int,
                              ) -> Array:
    """Return an (n_samples × K) responsibility matrix for the init method."""
    x_kmeans = x[:, None] if x.ndim == 1 else x
    r = np.zeros((n_samples, model.n_components), dtype=float)
    match model.init:
        case "k-means++":
            _, idx = kmeans_plusplus(
                x_kmeans, model.n_components, random_state=rng, sample_weight=sample_weight
            )
            r[idx, np.arange(model.n_components)] = 1.0
            return r

        case "k-means":
            labels = KMeans(
                n_clusters=model.n_components,
                init="random",
                n_init=10,
                max_iter=10,
                random_state=rng,
                # sample_weight=sample_weight,
            ).fit_predict(x_kmeans)
            r[np.arange(n_samples), labels] = 1.0  # as in hard clustering
            return r

        case "random_from_data":
            idx = rng.choice(n_samples, model.n_components, replace=False)
            r[idx, np.arange(model.n_components)] = 1.0
            return r

        case "random":
            # Empty -> triggers fallback to random responsibilities
            return r

        case _:
            raise ValueError(f"Unknown init method: {model.init!r}")

def fit_from_initial_posteriors(model: "MixtureModel",
                                x: Array,
                                post: Array,
                                sample_weight: Array,
                                eps: float,
                                ) -> None:
    """Fit each component and update mixture weights once."""
    for j, dist in enumerate(model.components):
        dist.fit(x, sample_weight=post[:, j] * sample_weight)
    model.weights = post.sum(axis=0) + eps # setter will normalize

def initialize_model(model: "MixtureModel",
                    x: Array,
                    sample_weight: Array,
                    ) -> None:

        x = np.asarray(x, dtype=float)
        n_samples = x.shape[0]
        eps = 10 * np.finfo(float).eps

        # Build the posterior matrix for the chosen strategy
        post = build_initial_posteriors(model, x, sample_weight, model.rng, n_samples)

        # Fallback to fully random soft responsibilities when initialization
        # is degenerate (empty clusters or fewer than 2 assigned samples).
        n_assigned = (post > 0).sum(axis=0)
        if (post.sum(axis=0) == 0).any() or (n_assigned < 2).any():
            post = model.rng.random((n_samples, model.n_components))

        # Fit components & weights once
        fit_from_initial_posteriors(model, x, post, sample_weight, eps)
