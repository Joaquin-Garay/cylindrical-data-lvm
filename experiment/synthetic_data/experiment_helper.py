"""Synthetic data experiment. Helper module."""

import numpy as np
import matplotlib.pyplot as plt

import soccer_pattern_recognition as spr

D_GAUSS = 3
D_VMF = 3

def _set_axes_equal_3d(ax):
    xlim = np.array(ax.get_xlim3d())
    ylim = np.array(ax.get_ylim3d())
    zlim = np.array(ax.get_zlim3d())
    center = np.array([xlim.mean(), ylim.mean(), zlim.mean()])
    radius = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


def _safe_weights(mixture):
    try:
        w = np.asarray(mixture.weights, dtype=float)
        if w.ndim != 1:
            return None
        return w
    except Exception:
        return None


def plot_cylindrical_components(
    model_or_components,
    *,
    n_std=1.5,
    arrow_scale=1.0,
    normalize_kappa=True,
    alpha=0.22,
    n_ellipsoid=36,
    ax=None,
):
    """
    Plot Cylindrical components in 3D:
    - ellipsoid centered at mu_gauss with shape from cond_cov
    - arrow from center in direction mu_vmf with length proportional to kappa
    """
    # Accept either a MixtureModel-like object or a plain list of components
    if hasattr(model_or_components, "components"):
        components = list(model_or_components.components)
        weights = getattr(model_or_components, "weights", None)
    else:
        components = list(model_or_components)
        weights = None

    if len(components) == 0:
        raise ValueError("No components to plot.")

    # Validate 3D setting
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

    # Unit sphere grid
    u = np.linspace(0.0, 2.0 * np.pi, n_ellipsoid)
    v = np.linspace(0.0, np.pi, n_ellipsoid)
    uu, vv = np.meshgrid(u, v)
    sphere = np.stack(
        [np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)],
        axis=-1,
    )  # (..., 3)

    cmap = plt.cm.tab10

    for i, comp in enumerate(components):
        color = cmap(i % 10)

        mu = comp.mu_gauss
        Sigma = comp.cond_cov
        evals, evecs = np.linalg.eigh(Sigma)
        evals = np.clip(evals, 1e-12, None)
        radii = n_std * np.sqrt(evals)

        # Ellipsoid: sphere -> scale by radii -> rotate by eigenvectors -> shift by mean
        ell = (sphere * radii) @ evecs.T + mu
        ax.plot_surface(
            ell[..., 0], ell[..., 1], ell[..., 2],
            rstride=1, cstride=1, linewidth=0.0, antialiased=True,
            alpha=alpha, color=color
        )

        # Center
        ax.scatter(mu[0], mu[1], mu[2], color=color, s=40)

        # vMF arrow
        direction = comp.vmf.mu  # unit vector
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
    ax.set_title("Cylindrical mixture parameters")
    _set_axes_equal_3d(ax)
    return fig, ax


def plot_mom_components(
    mom_or_layers,
    *,
    n_std=1.5,
    arrow_scale=1.0,
    normalize_kappa=True,
    scale_by_layer2_weight=True,
    alpha=0.22,
    n_ellipsoid=36,
    annotate_arrows=True,
    ax=None,
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

    # Validate 3D setting for layer 1 (Gaussian) and layer 2 (vMF)
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

    # Unit sphere grid
    u = np.linspace(0.0, 2.0 * np.pi, n_ellipsoid)
    v = np.linspace(0.0, np.pi, n_ellipsoid)
    uu, vv = np.meshgrid(u, v)
    sphere = np.stack(
        [np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)],
        axis=-1,
    )  # (..., 3)

    l1_weights = _safe_weights(layer1_mixture)
    cmap = plt.cm.tab10

    for i, (gauss, vmf_mix) in enumerate(zip(layer1_components, layer2_mixtures)):
        color = cmap(i % 10)
        mean, cov = gauss.params

        # Ellipsoid: sphere -> scale by radii -> rotate by eigenvectors -> shift by mean
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
        ax.text(mean[0], mean[1], mean[2],label, color=color)

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
            #{i}:{j} (w2={w2:.2f})
            if annotate_arrows:
                tip = mean + vec
                ax.text(tip[0], tip[1], tip[2], label, color=color)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Two-layer MoM parameters")
    _set_axes_equal_3d(ax)
    return fig, ax


def plot_cross_cov_matrices(
    components,
    cmap="coolwarm",
    max_cols=4,
    show=True,
    return_handles=False,
):
    mats = [np.asarray(c.cross_cov, dtype=float) for c in components]
    if len(mats) == 0:
        raise ValueError("No components to plot.")
    if not isinstance(max_cols, int) or max_cols < 1:
        raise ValueError("max_cols must be an integer >= 1.")
    for i, m in enumerate(mats):
        if m.ndim != 2:
            raise ValueError(
                f"Component {i} cross_cov must be 2D, got shape {m.shape}."
            )

    vmax = max(float(np.abs(m).max()) for m in mats)
    n = len(mats)
    n_cols = min(max_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    im = None
    for k, M in enumerate(mats):
        row, col = divmod(k, n_cols)
        ax = axes[row, col]
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(f"Component {k}")
        ax.set_xlabel("vMF dim")
        ax.set_ylabel("Gaussian dim")
        ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(M.shape[0]))

    for k in range(n, n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row, col].axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="cross_cov value")

    if show:
        plt.show()

    if return_handles:
        return fig, axes
    return None

def mom_builder_3d(n_layer1_components, n_layer2_components, init_layer1, init_layer2, rng):
    layer1_mixture = spr.MixtureModel(
        [spr.MultivariateGaussian(D_GAUSS) for _ in range(n_layer1_components)],
        init=init_layer1,
        rng=rng,
    )
    layer2_mixtures = [
        spr.MixtureModel(
            [spr.VonMisesFisher(D_VMF) for _ in range(n_layer2_components)],
            init=init_layer2,
            rng=rng,
        )
        for _ in range(n_layer1_components)
    ]
    return spr.TwoLayerMoM(layer1_mixture=layer1_mixture, layer2_mixtures=layer2_mixtures)

def mom_iso_builder_3d(n_layer1_components, n_layer2_components, init_layer1, init_layer2, rng):
    layer1_mixture = spr.MixtureModel(
        [spr.MultivariateGaussian(D_GAUSS) for _ in range(n_layer1_components)],
        init=init_layer1,
        rng=rng,
    )
    layer2_mixtures = [
        spr.MixtureModel(
            [spr.VonMisesFisher(D_VMF) for _ in range(n_layer2_components)],
            init=init_layer2,
            rng=rng,
        )
        for _ in range(n_layer1_components)
    ]
    return spr.IsolatedTwoLayerMoM(layer1_mixture=layer1_mixture, layer2_mixtures=layer2_mixtures)


def cylindrical_mixture_builder_3d(n_components, init, rng):
    """Builder for 3D cylindrical mixtures used by BIC grid-search calibration."""
    return spr.MixtureModel(
        [spr.Cylindrical(d_gauss=D_GAUSS, d_vmf=D_VMF) for _ in range(n_components)],
        init=init,
        rng=rng,
    )

def ind_cylindrical_mixture_builder_3d(n_components, init, rng):
    """Builder for 3D cylindrical mixtures used by BIC grid-search calibration."""
    return spr.MixtureModel(
        [spr.IndCylindrical(d_gauss=D_GAUSS, d_vmf=D_VMF) for _ in range(n_components)],
        init=init,
        rng=rng,
    )
