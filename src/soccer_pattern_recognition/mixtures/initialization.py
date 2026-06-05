"""Initialization methods for MixtureModel objects."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans

from ..clustering import cylindrical_kmeans, spherical_kmeans
from ..core.types import Array
from ..distributions import Cylindrical
from ..distributions.expfam import IndCylindrical, VonMises, VonMisesFisher

if TYPE_CHECKING:
    from ..mixtures.mixture import MixtureModel


def _all_spherical_components(model: "MixtureModel") -> bool:
    return all(isinstance(comp, (VonMises, VonMisesFisher)) for comp in model.components)


def _all_cylindrical_components(model: "MixtureModel") -> bool:
    return all(isinstance(comp, (Cylindrical, IndCylindrical)) for comp in model.components)


def _split_cylindrical_blocks(model: "MixtureModel", x: Array) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Cylindrical k-means initialization expects x with shape (n_samples, n_features).")

    first = model.components[0]
    d_gauss = int(first.d_gauss)
    d_vmf = int(first.d_vmf)
    expected_total = d_gauss + d_vmf
    if x.shape[1] != expected_total:
        raise ValueError(
            "Cylindrical k-means initialization dimension mismatch: "
            f"expected {expected_total} features (= d_gauss {d_gauss} + d_vmf {d_vmf}), got {x.shape[1]}."
        )

    for idx, comp in enumerate(model.components[1:], start=1):
        comp_d_gauss = int(comp.d_gauss)
        comp_d_vmf = int(comp.d_vmf)
        if comp_d_gauss != d_gauss or comp_d_vmf != d_vmf:
            raise ValueError(
                "All cylindrical mixture components must share the same (d_gauss, d_vmf). "
                f"Component 0 has ({d_gauss}, {d_vmf}), component {idx} has ({comp_d_gauss}, {comp_d_vmf})."
            )

    return x[:, :d_gauss], x[:, d_gauss:]


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
        case "k-means":
            if _all_spherical_components(model):
                labels, _, _ = spherical_kmeans(
                    x_kmeans,
                    n_clusters=model.n_components,
                    n_init=10,
                    max_iter=10,
                    rng=rng,
                )
            elif _all_cylindrical_components(model):
                x_euclid, x_spherical = _split_cylindrical_blocks(model, x_kmeans)
                labels, _, _, _ = cylindrical_kmeans(
                    x_euclid,
                    x_spherical,
                    lambda_ = 1.0,
                    n_clusters=model.n_components,
                    n_init=10,
                    max_iter=10,
                    rng=rng,
                )
            else:
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
