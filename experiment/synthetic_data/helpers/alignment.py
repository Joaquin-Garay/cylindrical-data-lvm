"""Component alignment helpers for synthetic data experiments."""

import numpy as np

import cyl_lvm as clvm

from .common import _safe_weights


def align_cylindrical_mixture_to_generator(
    model,
    generator,
    *,
    return_comparison=False,
):
    """
    Reorder a Cylindrical/IndCylindrical mixture in place to match a generator.

    The matching is exactly the one used by ``clvm.cylmix_comparison``: it
    minimizes total ``AbstractCylindrical.params_diff`` across components. After
    reordering, ``model.components[j]`` is the fitted component matched to
    ``generator.components[j]``. Initialized mixture weights are reordered with
    their components.
    """
    comparison = clvm.cylmix_comparison(model, generator)
    matching = list(comparison["matching"])
    components = model.components
    n_components = len(matching)

    aligned_components = [None] * n_components
    model_order = [None] * n_components
    old_weights = _safe_weights(model)
    aligned_weights = None if old_weights is None else np.empty(n_components, dtype=float)

    for model_idx, generator_idx in matching:
        if aligned_components[generator_idx] is not None:
            raise RuntimeError(
                "Invalid component matching: multiple model components matched "
                f"generator component {generator_idx}."
            )
        aligned_components[generator_idx] = components[model_idx]
        model_order[generator_idx] = model_idx
        if aligned_weights is not None:
            aligned_weights[generator_idx] = old_weights[model_idx]

    missing = [idx for idx, component in enumerate(aligned_components) if component is None]
    if missing:
        raise RuntimeError(
            "Invalid component matching: no model component matched generator "
            f"components {missing}."
        )

    try:
        components[:] = aligned_components
    except TypeError as exc:
        raise TypeError("model.components must be mutable for in-place reordering.") from exc

    if aligned_weights is not None:
        model.weights = aligned_weights

    if return_comparison:
        comparison = dict(comparison)
        comparison["model_order_for_generator"] = list(model_order)
        return model, comparison

    return model
