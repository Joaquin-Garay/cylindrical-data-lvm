"""Clustering quality metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ..core.types import Array, ArrayLike
from ..distributions import AbstractCylindrical

if TYPE_CHECKING:
    from ..mixtures import MixtureModel

__all__ = ["adjusted_rand_index", "ari", "cylmix_comparison"]

_STRING_LABEL_TYPES = (str, bytes, np.str_, np.bytes_)


def _as_label_vector(labels: ArrayLike, name: str) -> Array:
    arr = np.asarray(labels)
    object_arr = np.asarray(labels, dtype=object)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one label.")
    if arr.dtype.kind == "c":
        raise TypeError(f"{name} cannot contain complex labels.")

    for label in object_arr:
        if label is None:
            raise ValueError(f"{name} cannot contain missing labels.")
        if isinstance(label, (float, np.floating)):
            if not np.isfinite(label):
                raise ValueError(f"{name} cannot contain missing or infinite labels.")
            if not np.equal(label, np.rint(label)):
                raise ValueError(
                    f"{name} must contain discrete labels; got non-integer float values."
                )

    if arr.dtype.kind in {"U", "S"} and any(
        not isinstance(label, _STRING_LABEL_TYPES) for label in object_arr
    ):
        raise TypeError(
            f"{name} must use a consistent label type; got string labels mixed "
            "with non-string labels."
        )

    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} cannot contain missing or infinite labels.")
        if not np.equal(arr, np.rint(arr)).all():
            raise ValueError(
                f"{name} must contain discrete labels; got non-integer float values."
            )

    return arr


def adjusted_rand_index(labels_1: ArrayLike, labels_2: ArrayLike) -> float:
    """Return the adjusted Rand index between two cluster labelings.

    The metric is invariant to label permutations. A score of ``1.0`` means
    perfect agreement, while scores near ``0.0`` are expected for independent
    random labelings.
    """
    labels_1_arr = _as_label_vector(labels_1, "labels_1")
    labels_2_arr = _as_label_vector(labels_2, "labels_2")

    if labels_1_arr.shape[0] != labels_2_arr.shape[0]:
        raise ValueError(
            "labels_1 and labels_2 must contain the same number of labels; "
            f"got {labels_1_arr.shape[0]} and {labels_2_arr.shape[0]}."
        )

    return float(adjusted_rand_score(labels_1_arr, labels_2_arr))


def ari(labels_1: ArrayLike, labels_2: ArrayLike) -> float:
    """Alias for :func:`adjusted_rand_index`."""
    return adjusted_rand_index(labels_1, labels_2)


def _validate_cylmix(cylmix: "MixtureModel", *, name: str) -> list[AbstractCylindrical]:
    if not hasattr(cylmix, "components") or not hasattr(cylmix, "n_components"):
        raise TypeError(f"{name} must be a mixture model with components and n_components.")

    components = list(cylmix.components)
    if int(cylmix.n_components) != len(components):
        raise ValueError(
            f"{name}.n_components does not match len({name}.components): "
            f"{cylmix.n_components} != {len(components)}."
        )
    if len(components) == 0:
        raise ValueError(f"{name} must contain at least one component.")

    for idx, component in enumerate(components):
        if not isinstance(component, AbstractCylindrical):
            raise TypeError(
                f"{name}.components[{idx}] must be Cylindrical or IndCylindrical, got "
                f"{type(component).__name__}."
            )

    return components


def _validate_same_cylindrical_dimensions(
    components_1: list[AbstractCylindrical],
    components_2: list[AbstractCylindrical],
) -> None:
    d_gauss = components_1[0].d_gauss
    d_vmf = components_1[0].d_vmf
    for name, components in {
        "cylmix_1": components_1,
        "cylmix_2": components_2,
    }.items():
        for idx, component in enumerate(components):
            if (component.d_gauss, component.d_vmf) != (d_gauss, d_vmf):
                raise ValueError(
                    "All cylindrical components must have the same dimensions; "
                    f"cylmix_1.components[0] has ({d_gauss}, {d_vmf}) but "
                    f"{name}.components[{idx}] has "
                    f"({component.d_gauss}, {component.d_vmf})."
                )


def _minimum_cost_matching(cost: Array) -> list[tuple[int, int]]:
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be a square matrix.")
    if not np.all(np.isfinite(cost)):
        raise ValueError("cost contains non-finite values.")

    n_components = cost.shape[0]
    memo: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}

    def solve(row: int, used_cols_mask: int) -> tuple[float, tuple[int, ...]]:
        if row == n_components:
            return 0.0, ()

        key = (row, used_cols_mask)
        if key in memo:
            return memo[key]

        best_cost = np.inf
        best_cols: tuple[int, ...] = ()
        for col in range(n_components):
            if used_cols_mask & (1 << col):
                continue
            remaining_cost, remaining_cols = solve(row + 1, used_cols_mask | (1 << col))
            candidate_cost = float(cost[row, col]) + remaining_cost
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_cols = (col, *remaining_cols)

        memo[key] = best_cost, best_cols
        return memo[key]

    _, cols = solve(0, 0)
    return [(row, int(col)) for row, col in enumerate(cols)]


def cylmix_comparison(cylmix_1: "MixtureModel", cylmix_2: "MixtureModel") -> dict:
    """Compare two cylindrical mixtures after resolving component label switching.

    Component matching is based on ``AbstractCylindrical.params_diff``.
    """
    components_1 = _validate_cylmix(cylmix_1, name="cylmix_1")
    components_2 = _validate_cylmix(cylmix_2, name="cylmix_2")

    if len(components_1) != len(components_2):
        raise ValueError(
            "cylmix_1 and cylmix_2 must have the same number of clusters; "
            f"got {len(components_1)} and {len(components_2)}."
        )

    _validate_same_cylindrical_dimensions(components_1, components_2)

    n_components = len(components_1)
    pairwise_params_diff = np.empty((n_components, n_components), dtype=float)

    for i, component_1 in enumerate(components_1):
        for j, component_2 in enumerate(components_2):
            pairwise_params_diff[i, j] = AbstractCylindrical.params_diff(
                component_1,
                component_2,
            )

    matching = _minimum_cost_matching(pairwise_params_diff)

    comparison = {
        "cond_cov": [],
        "cross_cov": [],
        "mu_gauss": [],
        "mu_vmf": [],
        "kappa_vmf": [],
        "params_diff": [],
        "matching": matching,
        "pairwise_params_diff": pairwise_params_diff,
    }

    for i, j in matching:
        comparison["params_diff"].append(round(float(pairwise_params_diff[i, j]), 5))
        comparison["cond_cov"].append(
            round(AbstractCylindrical.cond_cov_diff(components_1[i], components_2[j]), 5)
        )
        comparison["cross_cov"].append(
            round(AbstractCylindrical.cross_cov_diff(components_1[i], components_2[j]), 5)
        )
        comparison["mu_gauss"].append(
            round(AbstractCylindrical.mu_gauss_diff(components_1[i], components_2[j]), 5)
        )
        comparison["mu_vmf"].append(
            round(AbstractCylindrical.mu_vmf_diff(components_1[i], components_2[j]), 5)
        )
        comparison["kappa_vmf"].append(
            round(AbstractCylindrical.kappa_diff(components_1[i], components_2[j]), 5)
        )

    return comparison
