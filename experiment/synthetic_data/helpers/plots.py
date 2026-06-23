"""Plotting helpers for synthetic data experiments."""

import numpy as np
import matplotlib.pyplot as plt
from cyl_lvm.mixtures import MixtureModel

from .common import (
    _extract_cross_corr_matrices,
    _extract_cylindrical_gaussian_params,
    _is_mom_like,
    _mom_cross_corr_matrices,
    _safe_weights,
)


def _set_axes_equal_3d(ax):
    xlim = np.array(ax.get_xlim3d())
    ylim = np.array(ax.get_ylim3d())
    zlim = np.array(ax.get_zlim3d())
    center = np.array([xlim.mean(), ylim.mean(), zlim.mean()])
    radius = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


def _set_axes_equal_2d(ax):
    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())
    center = np.array([xlim.mean(), ylim.mean()])
    radius = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def _sync_2d_axes(axes):
    axes = np.asarray(axes).ravel()
    xlim = np.array([ax.get_xlim() for ax in axes], dtype=float)
    ylim = np.array([ax.get_ylim() for ax in axes], dtype=float)
    center = np.array([xlim.mean(), ylim.mean()])
    radius = 0.5 * max(
        xlim[:, 1].max() - xlim[:, 0].min(),
        ylim[:, 1].max() - ylim[:, 0].min(),
    )

    for ax in axes:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal", adjustable="box")


def _plot_cross_corr_matrix_grid(
    mats,
    *,
    fig=None,
    axes=None,
    cmap="coolwarm",
    max_cols=4,
    title_prefix="Component",
):
    if len(mats) == 0:
        raise ValueError("No matrices to plot.")
    if not isinstance(max_cols, int) or max_cols < 1:
        raise ValueError("max_cols must be an integer >= 1.")

    n = len(mats)
    n_cols = min(max_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    if axes is None:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3 * n_cols, 2 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
    else:
        axes = np.asarray(axes)
        if axes.ndim != 2:
            raise ValueError("axes must be a 2D array.")
        n_rows, n_cols = axes.shape
        if n > n_rows * n_cols:
            raise ValueError(f"axes must have at least {n} slots.")
        fig = axes.ravel()[0].figure if fig is None else fig

    im = None
    for k, M in enumerate(mats):
        row, col = divmod(k, n_cols)
        ax = axes[row, col]
        im = ax.imshow(M, cmap=cmap, vmin=-1.0, vmax=1.0)
        ax.set_title(f"{title_prefix} {k}")
        ax.set_xlabel("vMF dim")
        ax.set_ylabel("Gaussian dim")
        ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(M.shape[0]))

    for k in range(n, n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row, col].axis("off")

    return fig, axes, im


def _mixture_weights(mixture, *, name):
    if not isinstance(mixture, MixtureModel):
        raise ValueError(f"{name} must be a MixtureModel instance.")

    n_components = len(mixture.components)
    if n_components == 0:
        raise ValueError(f"{name} must have at least one component.")

    weights = _safe_weights(mixture)
    if weights is None:
        raise ValueError(f"{name} weights are not initialized or are not a 1D vector.")
    if weights.size != n_components:
        raise ValueError(
            f"{name} weights length must match the number of components; "
            f"got {weights.size} weights and {n_components} components."
        )
    return weights


def _plot_mixing_weights_bar(ax, weights, *, title):
    x = np.arange(weights.size)
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in x]

    ax.bar(x, weights, color=colors)
    ax.set_title(str(title))
    ax.set_xlabel("Component")
    ax.set_ylabel("Mixing weight")
    ax.set_xticks(x)
    ax.set_ylim(0.0, max(1.0, float(np.max(weights)) * 1.1))
    ax.grid(axis="y", alpha=0.2)


def plot_mixing_weights_model_vs_generator(
    model,
    generator,
    *,
    model_title=None,
    generator_title="Generator",
    title=None,
    figsize=(8, 3),
    share_ylim=True,
):
    """
    Plot model and generator mixing weights in a 1x2 bar-plot layout.

    Bar colors follow the same tab10 component order used by the parameter
    plotters.
    """
    if model_title is None:
        model_title = "Model" if title is None else str(title)

    model_weights = _mixture_weights(model, name="model")
    generator_weights = _mixture_weights(generator, name="generator")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    left_ax, right_ax = axes[0]

    _plot_mixing_weights_bar(left_ax, model_weights, title=model_title)
    _plot_mixing_weights_bar(right_ax, generator_weights, title=generator_title)

    if share_ylim:
        ymax = max(left_ax.get_ylim()[1], right_ax.get_ylim()[1])
        left_ax.set_ylim(0.0, ymax)
        right_ax.set_ylim(0.0, ymax)

    return fig, axes[0]


def plot_cylindrical_components_3d(
    model_or_components,
    *,
    n_std=1.5,
    arrow_scale=1.0,
    normalize_kappa=True,
    alpha=0.22,
    n_ellipsoid=36,
    ax=None,
    title=None,
):
    """
    Plot Cylindrical components in 3D:
    - ellipsoid centered at mu_gauss with shape from unconditional Gaussian covariance
    - arrow from center in direction mu_vmf with length proportional to kappa
    """
    if hasattr(model_or_components, "components"):
        components = list(model_or_components.components)
    else:
        components = list(model_or_components)

    if len(components) == 0:
        raise ValueError("No components to plot.")

    for c in components:
        if c.d_gauss != 3 or c.d_vmf != 3:
            raise ValueError("This plotter expects d_gauss=3 and d_vmf=3 for all components.")

    kappas = np.array([c.vmf.kappa for c in components], dtype=float)
    kappa_ref = np.max(kappas) if normalize_kappa else 1.0
    if kappa_ref <= 0:
        kappa_ref = 1.0

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    u = np.linspace(0.0, 2.0 * np.pi, n_ellipsoid)
    v = np.linspace(0.0, np.pi, n_ellipsoid)
    uu, vv = np.meshgrid(u, v)
    sphere = np.stack(
        [np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)],
        axis=-1,
    )

    cmap = plt.cm.tab10

    for i, comp in enumerate(components):
        color = cmap(i % 10)

        mu, Sigma = _extract_cylindrical_gaussian_params(comp)
        evals, evecs = np.linalg.eigh(Sigma)
        evals = np.clip(evals, 1e-12, None)
        radii = n_std * np.sqrt(evals)

        ell = (sphere * radii) @ evecs.T + mu
        ax.plot_surface(
            ell[..., 0], ell[..., 1], ell[..., 2],
            rstride=1, cstride=1, linewidth=0.0, antialiased=True,
            alpha=alpha, color=color
        )

        ax.scatter(mu[0], mu[1], mu[2], color=color, s=40)

        direction = comp.vmf.mu
        length = arrow_scale * (comp.vmf.kappa / kappa_ref)
        vec = length * direction
        ax.quiver(
            mu[0], mu[1], mu[2],
            vec[0], vec[1], vec[2],
            color=color, linewidth=2.0, arrow_length_ratio=0.15
        )

        label = ""
        ax.text(mu[0], mu[1], mu[2], label, color=color)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Cylindrical mixture parameters" if title is None else str(title))
    _set_axes_equal_3d(ax)
    return fig, ax


def plot_cylindrical_components_2d(
    model_or_components,
    *,
    n_std=1.5,
    arrow_scale=1.0,
    normalize_kappa=True,
    alpha=0.22,
    n_ellipse=200,
    ax=None,
    title=None,
):
    """
    Plot Cylindrical components in 2D:
    - ellipse centered at mu_gauss with shape from unconditional Gaussian covariance
    - arrow from center in direction mu_vmf with length proportional to kappa
    """
    if hasattr(model_or_components, "components"):
        components = list(model_or_components.components)
    else:
        components = list(model_or_components)

    if len(components) == 0:
        raise ValueError("No components to plot.")

    for c in components:
        if c.d_gauss != 2 or c.d_vmf != 2:
            raise ValueError("This plotter expects d_gauss=2 and d_vmf=2 for all components.")

    kappas = np.array([c.vmf.kappa for c in components], dtype=float)
    kappa_ref = np.max(kappas) if normalize_kappa else 1.0
    if kappa_ref <= 0:
        kappa_ref = 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    theta = np.linspace(0.0, 2.0 * np.pi, n_ellipse)
    circle = np.column_stack((np.cos(theta), np.sin(theta)))

    cmap = plt.cm.tab10

    for i, comp in enumerate(components):
        color = cmap(i % 10)

        mu, Sigma = _extract_cylindrical_gaussian_params(comp)
        evals, evecs = np.linalg.eigh(Sigma)
        evals = np.clip(evals, 1e-12, None)
        radii = n_std * np.sqrt(evals)

        ell = (circle * radii) @ evecs.T + mu
        ax.fill(ell[:, 0], ell[:, 1], color=color, alpha=alpha, linewidth=0.0)
        ax.plot(ell[:, 0], ell[:, 1], color=color, linewidth=2.0)

        ax.scatter(mu[0], mu[1], color=color, s=40, zorder=3)

        direction = comp.vmf.mu
        length = arrow_scale * (comp.vmf.kappa / kappa_ref)
        vec = length * direction
        ax.quiver(
            mu[0], mu[1],
            vec[0], vec[1],
            angles="xy", scale_units="xy", scale=1.0,
            color=color, linewidth=2.0, width=0.006, zorder=4
        )

        label = ""
        ax.text(mu[0], mu[1], label, color=color)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D cylindrical mixture parameters" if title is None else str(title))
    ax.grid(True, alpha=0.2)
    _set_axes_equal_2d(ax)
    return fig, ax


def plot_mom_components_3d(
    mom_or_layers,
    *,
    n_std=1.5,
    arrow_scale=1.0,
    normalize_kappa=True,
    scale_by_layer2_weight=False,
    alpha=0.22,
    n_ellipsoid=36,
    annotate_arrows=True,
    ax=None,
    title=None,
):
    """
    Plot TwoLayerMoM parameters in 3D:
    - layer-1 Gaussian ellipsoids
    - layer-2 vMF arrows from each parent Gaussian mean

    Accepts either:
    - a TwoLayerMoM-like object with `.layer1_mixture` and `.layer2_mixtures`
    - a tuple `(layer1_mixture, layer2_mixtures)`
    """
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

    for comp in layer1_components:
        mean, cov = comp.params
        if mean.shape != (3,) or cov.shape != (3, 3):
            raise ValueError("This plotter expects layer-1 Gaussian components in 3D.")

    all_kappas = []
    for mix in layer2_mixtures:
        if len(mix.components) == 0:
            raise ValueError("Each layer-2 mixture must have at least one component.")
        for vmf in mix.components:
            if getattr(vmf, "d", None) != 3:
                raise ValueError("This plotter expects layer-2 vMF components in 3D.")
            all_kappas.append(float(vmf.kappa))

    all_kappas = np.asarray(all_kappas, dtype=float)
    kappa_ref = np.max(all_kappas) if normalize_kappa else 1.0
    if not np.isfinite(kappa_ref) or kappa_ref <= 0.0:
        kappa_ref = 1.0

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    u = np.linspace(0.0, 2.0 * np.pi, n_ellipsoid)
    v = np.linspace(0.0, np.pi, n_ellipsoid)
    uu, vv = np.meshgrid(u, v)
    sphere = np.stack(
        [np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)],
        axis=-1,
    )

    cmap = plt.cm.tab10

    for i, (gauss, vmf_mix) in enumerate(zip(layer1_components, layer2_mixtures)):
        color = cmap(i % 10)
        mean, cov = gauss.params

        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 1e-12, None)
        radii = n_std * np.sqrt(evals)
        ell = (sphere * radii) @ evecs.T + mean
        ax.plot_surface(
            ell[..., 0], ell[..., 1], ell[..., 2],
            rstride=1, cstride=1, linewidth=0.0, antialiased=True,
            alpha=alpha, color=color
        )
        ax.scatter(mean[0], mean[1], mean[2], color=color, s=40)

        label = ""
        ax.text(mean[0], mean[1], mean[2], label, color=color)

        l2_weights = _safe_weights(vmf_mix)
        if l2_weights is None or l2_weights.size != len(vmf_mix.components):
            l2_weights = np.full(len(vmf_mix.components), 1.0 / len(vmf_mix.components))

        for j, (vmf, w2) in enumerate(zip(vmf_mix.components, l2_weights)):
            direction = vmf.mu
            length = arrow_scale * (float(vmf.kappa) / kappa_ref)
            if scale_by_layer2_weight:
                length *= float(w2)

            vec = length * direction
            ax.quiver(
                mean[0], mean[1], mean[2],
                vec[0], vec[1], vec[2],
                color=color,
                linewidth=1.0 + 1.5 * float(w2),
                alpha=0.45 + 0.5 * float(w2),
                arrow_length_ratio=0.15,
            )
            if annotate_arrows:
                tip = mean + vec
                ax.text(tip[0], tip[1], tip[2], label, color=color)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Two-layer MoM parameters" if title is None else str(title))
    _set_axes_equal_3d(ax)
    return fig, ax


def plot_mom_components_2d(
    mom_or_layers,
    *,
    n_std=1.5,
    arrow_scale=1.0,
    normalize_kappa=True,
    scale_by_layer2_weight=False,
    alpha=0.22,
    n_ellipse=200,
    annotate_arrows=True,
    ax=None,
    title=None,
):
    """
    Plot TwoLayerMoM parameters in 2D:
    - layer-1 Gaussian ellipses
    - layer-2 vMF arrows from each parent Gaussian mean

    Accepts either:
    - a TwoLayerMoM-like object with `.layer1_mixture` and `.layer2_mixtures`
    - a tuple `(layer1_mixture, layer2_mixtures)`
    """
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

    for comp in layer1_components:
        mean, cov = comp.params
        if mean.shape != (2,) or cov.shape != (2, 2):
            raise ValueError("This plotter expects layer-1 Gaussian components in 2D.")

    all_kappas = []
    for mix in layer2_mixtures:
        if len(mix.components) == 0:
            raise ValueError("Each layer-2 mixture must have at least one component.")
        for vmf in mix.components:
            if getattr(vmf, "d", None) != 2:
                raise ValueError("This plotter expects layer-2 vMF components in 2D.")
            all_kappas.append(float(vmf.kappa))

    all_kappas = np.asarray(all_kappas, dtype=float)
    kappa_ref = np.max(all_kappas) if normalize_kappa else 1.0
    if not np.isfinite(kappa_ref) or kappa_ref <= 0.0:
        kappa_ref = 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    theta = np.linspace(0.0, 2.0 * np.pi, n_ellipse)
    circle = np.column_stack((np.cos(theta), np.sin(theta)))

    cmap = plt.cm.tab10

    for i, (gauss, vmf_mix) in enumerate(zip(layer1_components, layer2_mixtures)):
        color = cmap(i % 10)
        mean, cov = gauss.params

        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 1e-12, None)
        radii = n_std * np.sqrt(evals)
        ell = (circle * radii) @ evecs.T + mean
        ax.fill(ell[:, 0], ell[:, 1], color=color, alpha=alpha, linewidth=0.0)
        ax.plot(ell[:, 0], ell[:, 1], color=color, linewidth=2.0)
        ax.scatter(mean[0], mean[1], color=color, s=40, zorder=3)

        label = ""
        ax.text(mean[0], mean[1], label, color=color)

        l2_weights = _safe_weights(vmf_mix)
        if l2_weights is None or l2_weights.size != len(vmf_mix.components):
            l2_weights = np.full(len(vmf_mix.components), 1.0 / len(vmf_mix.components))

        for j, (vmf, w2) in enumerate(zip(vmf_mix.components, l2_weights)):
            direction = vmf.mu
            length = arrow_scale * (float(vmf.kappa) / kappa_ref)
            if scale_by_layer2_weight:
                length *= float(w2)

            vec = length * direction
            ax.quiver(
                mean[0], mean[1],
                vec[0], vec[1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                linewidth=1.0 + 1.5 * float(w2),
                alpha=0.45 + 0.5 * float(w2),
                width=0.006,
                zorder=4,
            )
            if annotate_arrows:
                tip = mean + vec
                ax.text(tip[0], tip[1], label, color=color)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D two-layer MoM parameters" if title is None else str(title))
    ax.grid(True, alpha=0.2)
    _set_axes_equal_2d(ax)
    return fig, ax


def plot_model_vs_generator_2d(
    model,
    generator,
    *,
    title=None,
    model_title=None,
    generator_title="Generator",
    model_kind="auto",
    arrow_scale=None,
    model_arrow_scale=None,
    generator_arrow_scale=0.3,
    normalize_kappa=False,
    n_std=1.5,
    alpha=0.22,
    n_ellipse=200,
    scale_by_layer2_weight=False,
    annotate_arrows=True,
    share_limits=True,
    figsize=(10, 5),
):
    """
    Plot a fitted 2D model next to the cylindrical generator.

    The left panel accepts either a Cylindrical mixture or a TwoLayerMoM-like
    model. The right panel is always plotted as a Cylindrical mixture.
    """
    if model_kind == "auto":
        model_kind = "mom" if _is_mom_like(model) else "cylindrical"
    if model_kind not in {"cylindrical", "mom"}:
        raise ValueError("model_kind must be one of {'auto', 'cylindrical', 'mom'}.")
    if arrow_scale is not None and model_arrow_scale is not None:
        raise ValueError("Provide only one of arrow_scale or model_arrow_scale.")

    if model_arrow_scale is None:
        model_arrow_scale = arrow_scale
    if model_arrow_scale is None:
        model_arrow_scale = 0.1 if model_kind == "mom" else 0.3
    if model_title is None:
        model_title = "Model" if title is None else str(title)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    left_ax, right_ax = axes[0]

    if model_kind == "mom":
        plot_mom_components_2d(
            model,
            n_std=n_std,
            arrow_scale=model_arrow_scale,
            normalize_kappa=normalize_kappa,
            scale_by_layer2_weight=scale_by_layer2_weight,
            alpha=alpha,
            n_ellipse=n_ellipse,
            annotate_arrows=annotate_arrows,
            ax=left_ax,
            title=model_title,
        )
    else:
        plot_cylindrical_components_2d(
            model,
            n_std=n_std,
            arrow_scale=model_arrow_scale,
            normalize_kappa=normalize_kappa,
            alpha=alpha,
            n_ellipse=n_ellipse,
            ax=left_ax,
            title=model_title,
        )

    plot_cylindrical_components_2d(
        generator,
        n_std=n_std,
        arrow_scale=generator_arrow_scale,
        normalize_kappa=normalize_kappa,
        alpha=alpha,
        n_ellipse=n_ellipse,
        ax=right_ax,
        title=generator_title,
    )

    if share_limits:
        _sync_2d_axes([left_ax, right_ax])

    return fig, axes[0]


def plot_cross_corr_matrices(
    components,
    cmap="coolwarm",
    max_cols=4,
    show=True,
    return_handles=False,
):
    mats = _extract_cross_corr_matrices(components)
    fig, axes, im = _plot_cross_corr_matrix_grid(
        mats,
        cmap=cmap,
        max_cols=max_cols,
    )

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)

    if show:
        plt.show()

    if return_handles:
        return fig, axes

    plt.close(fig)
    return None


def plot_cross_corr_model_vs_generator(
    model,
    generator,
    *,
    model_title=None,
    generator_title="Generator",
    title=None,
    model_kind="auto",
    cmap="coolwarm",
    max_cols=None,
    show=True,
    return_handles=False,
):
    if model_kind == "auto":
        model_kind = "mom" if _is_mom_like(model) else "cylindrical"
    if model_kind not in {"cylindrical", "mom"}:
        raise ValueError("model_kind must be one of {'auto', 'cylindrical', 'mom'}.")

    if model_title is None:
        model_title = "Model" if title is None else str(title)

    model_mats = (
        _mom_cross_corr_matrices(model)
        if model_kind == "mom"
        else _extract_cross_corr_matrices(model, name="model.components")
    )
    generator_mats = _extract_cross_corr_matrices(generator, name="generator.components")

    max_count = max(len(model_mats), len(generator_mats))
    if max_cols is None:
        n_cols = max_count
    else:
        if not isinstance(max_cols, int) or max_cols < 1:
            raise ValueError("max_cols must be an integer >= 1 or None.")
        n_cols = min(max_cols, max_count)
    n_rows = int(np.ceil(max_count / n_cols))

    fig, axes = plt.subplots(
        2 * n_rows,
        n_cols,
        figsize=(3 * n_cols, 4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    model_axes = axes[:n_rows, :]
    generator_axes = axes[n_rows:, :]

    _, model_axes, model_im = _plot_cross_corr_matrix_grid(
        model_mats,
        fig=fig,
        axes=model_axes,
        cmap=cmap,
        max_cols=n_cols,
        title_prefix=f"{model_title} component",
    )
    _, generator_axes, generator_im = _plot_cross_corr_matrix_grid(
        generator_mats,
        fig=fig,
        axes=generator_axes,
        cmap=cmap,
        max_cols=n_cols,
        title_prefix=f"{generator_title} component",
    )

    fig.colorbar(
        model_im if model_im is not None else generator_im,
        ax=axes.ravel().tolist(),
        shrink=0.85,
    )

    if show:
        plt.show()

    if return_handles:
        return fig, axes

    plt.close(fig)
    return None
