"""Shared utilities for synthetic data experiments."""

import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / max(norm, 1e-12)
    else:
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.clip(norm, 1e-12, None)


def _safe_weights(mixture):
    try:
        w = np.asarray(mixture.weights, dtype=float)
        if w.ndim != 1:
            return None
        return w
    except Exception:
        return None


def _is_mom_like(model):
    return (
        hasattr(model, "layer1_mixture")
        and hasattr(model, "layer2_mixtures")
    ) or (
        isinstance(model, (tuple, list))
        and len(model) == 2
        and hasattr(model[0], "components")
    )


def _extract_cylindrical_gaussian_params(component):
    if hasattr(component, "mu_gauss") and hasattr(component, "unconditional_gauss_cov"):
        return (
            np.asarray(component.mu_gauss, dtype=float),
            np.asarray(component.unconditional_gauss_cov, dtype=float),
        )

    if hasattr(component, "mu_gauss") and hasattr(component, "cond_cov"):
        return np.asarray(component.mu_gauss, dtype=float), np.asarray(component.cond_cov, dtype=float)

    if hasattr(component, "gaussian"):
        mean, cov = component.gaussian.params
        return np.asarray(mean, dtype=float), np.asarray(cov, dtype=float)

    raise ValueError(
        "Unsupported component type for cylindrical plot: expected Cylindrical "
        "or IndCylindrical-like component."
    )


def _extract_cross_corr_matrices(model_or_components, *, name="components"):
    if hasattr(model_or_components, "components"):
        components = list(model_or_components.components)
    else:
        components = list(model_or_components)

    mats = []
    for c in components:
        if hasattr(c, "cross_corr"):
            mats.append(np.asarray(c.cross_corr, dtype=float))
        elif hasattr(c, "d_gauss") and hasattr(c, "d_vmf"):
            mats.append(np.zeros((c.d_gauss, c.d_vmf), dtype=float))
        else:
            raise ValueError(
                f"{name} must contain components with cross_corr or cylindrical dimensions."
            )
    if len(mats) == 0:
        raise ValueError(f"No {name} to plot.")
    for i, m in enumerate(mats):
        if m.ndim != 2:
            raise ValueError(
                f"{name}[{i}] cross_corr must be 2D, got shape {m.shape}."
            )
    return mats


def _mom_cross_corr_matrices(mom_or_layers):
    if hasattr(mom_or_layers, "layer1_mixture") and hasattr(mom_or_layers, "layer2_mixtures"):
        layer1_mixture = mom_or_layers.layer1_mixture
        layer2_mixtures = list(mom_or_layers.layer2_mixtures)
    elif (
        isinstance(mom_or_layers, (tuple, list))
        and len(mom_or_layers) == 2
    ):
        layer1_mixture = mom_or_layers[0]
        layer2_mixtures = list(mom_or_layers[1])
    else:
        raise ValueError(
            "Expected a TwoLayerMoM-like object or a tuple "
            "(layer1_mixture, layer2_mixtures)."
        )

    layer1_components = list(layer1_mixture.components)
    if len(layer1_components) == 0:
        raise ValueError("No layer-1 components to plot.")
    if len(layer2_mixtures) != len(layer1_components):
        raise ValueError("Number of layer-2 mixtures must match layer-1 components.")

    mats = []
    for i, (gauss, vmf_mix) in enumerate(zip(layer1_components, layer2_mixtures)):
        mean, _ = gauss.params
        if len(vmf_mix.components) == 0:
            raise ValueError("Each layer-2 mixture must have at least one component.")
        d_gauss = np.asarray(mean, dtype=float).shape[0]
        d_vmf = getattr(vmf_mix.components[0], "d", None)
        if d_vmf is None:
            raise ValueError(f"layer2_mixtures[{i}] components must expose dimension d.")
        mats.append(np.zeros((d_gauss, int(d_vmf)), dtype=float))
    return mats
