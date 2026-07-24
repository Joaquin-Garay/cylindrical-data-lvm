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


def sample_noisy_train_test(
    generator,
    n,
    n_train,
    d_gauss,
    d_vmf=None,
    *,
    rng=None,
    noise_scale=0.15,
    min_std=1e-8,
):
    """
    Sample a mixed Euclidean/directional dataset, add noise, and split it.

    The first block is treated as Euclidean. The second block is normalized
    with ``unit`` before returning ``x`` and after adding noise for ``x_noise``.
    """
    if not isinstance(n, (int, np.integer)) or int(n) < 2:
        raise ValueError("n must be an integer >= 2.")
    n = int(n)

    if not isinstance(n_train, (int, np.integer)):
        raise ValueError("n_train must be an integer.")
    n_train = int(n_train)
    if n_train < 0 or n_train > n:
        raise ValueError(f"n_train must be between 0 and n={n}; got {n_train}.")

    if not isinstance(d_gauss, (int, np.integer)) or int(d_gauss) <= 0:
        raise ValueError("d_gauss must be a positive integer.")
    d_gauss = int(d_gauss)

    noise_scale = float(noise_scale)
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative.")
    min_std = float(min_std)
    if min_std <= 0.0:
        raise ValueError("min_std must be positive.")

    sample_kwargs = {"return_labels": True}
    if rng is not None:
        sample_kwargs["rng"] = rng
    x, labels = generator.sample(n, **sample_kwargs)
    x = np.asarray(x, dtype=float).copy()
    labels = np.asarray(labels)
    if x.ndim != 2:
        raise ValueError("generator.sample must return a 2D sample array.")
    if x.shape[0] != n:
        raise ValueError(f"Expected {n} samples; got {x.shape[0]}.")

    if d_vmf is None:
        d_vmf = x.shape[1] - d_gauss
    if not isinstance(d_vmf, (int, np.integer)) or int(d_vmf) <= 0:
        raise ValueError("d_vmf must be a positive integer.")
    d_vmf = int(d_vmf)
    if d_gauss + d_vmf != x.shape[1]:
        raise ValueError(
            "d_gauss + d_vmf must match the sample dimension; "
            f"got {d_gauss} + {d_vmf} for dimension {x.shape[1]}."
        )

    noise_rng = rng if rng is not None else np.random.default_rng()
    noise_g = noise_rng.normal(
        0.0,
        noise_scale * np.maximum(x[:, :d_gauss].std(axis=0, ddof=1), min_std),
        size=(n, d_gauss),
    )
    noise_v = noise_rng.normal(
        0.0,
        noise_scale * np.maximum(x[:, d_gauss:].std(axis=0, ddof=1), min_std),
        size=(n, d_vmf),
    )

    x[:, d_gauss:] = unit(x[:, d_gauss:])
    x_noise = np.concatenate(
        (
            noise_g + x[:, :d_gauss],
            unit(noise_v + x[:, d_gauss:]),
        ),
        axis=1,
    )

    return {
        "x": x,
        "x_noise": x_noise,
        "labels": labels,
        "noise_g": noise_g,
        "noise_v": noise_v,
        "x_train": x[:n_train],
        "x_test": x[n_train:],
        "x_train_noise": x_noise[:n_train],
        "x_test_noise": x_noise[n_train:],
        "labels_train": labels[:n_train],
        "labels_test": labels[n_train:],
    }


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
        return (
            np.asarray(component.mu_gauss, dtype=float),
            np.asarray(component.cond_cov, dtype=float),
        )

    if hasattr(component, "gaussian"):
        mean, cov = component.gaussian.params
        return np.asarray(mean, dtype=float), np.asarray(cov, dtype=float)

    raise ValueError(
        "Unsupported component type for cylindrical plot: expected Cylindrical "
        "or IndCylindrical-like component."
    )


