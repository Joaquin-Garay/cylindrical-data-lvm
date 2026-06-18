"""Synthetic data experiment. Helper module."""

import numpy as np
import matplotlib.pyplot as plt

import cyl_lvm as clvm

def unit(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / max(norm, 1e-12)
    else:
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.clip(norm, 1e-12, None)


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


def _safe_weights(mixture):
    try:
        w = np.asarray(mixture.weights, dtype=float)
        if w.ndim != 1:
            return None
        return w
    except Exception:
        return None


def _extract_cylindrical_gaussian_params(component):
    # Cylindrical: direct fields
    if hasattr(component, "mu_gauss") and hasattr(component, "cond_cov"):
        return np.asarray(component.mu_gauss, dtype=float), np.asarray(component.cond_cov, dtype=float)

    # IndCylindrical: nested Gaussian submodel
    if hasattr(component, "gaussian"):
        mean, cov = component.gaussian.params
        return np.asarray(mean, dtype=float), np.asarray(cov, dtype=float)

    raise ValueError(
        "Unsupported component type for cylindrical plot: expected Cylindrical "
        "or IndCylindrical-like component."
    )


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
    - ellipsoid centered at mu_gauss with shape from cond_cov
    - arrow from center in direction mu_vmf with length proportional to kappa
    """
    # Accept either a MixtureModel-like object or a plain list of components
    if hasattr(model_or_components, "components"):
        components = list(model_or_components.components)
    else:
        components = list(model_or_components)

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

        mu, Sigma = _extract_cylindrical_gaussian_params(comp)
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
    - ellipse centered at mu_gauss with shape from cond_cov
    - arrow from center in direction mu_vmf with length proportional to kappa
    """
    # Accept either a MixtureModel-like object or a plain list of components
    if hasattr(model_or_components, "components"):
        components = list(model_or_components.components)
    else:
        components = list(model_or_components)

    if len(components) == 0:
        raise ValueError("No components to plot.")

    # Validate 2D setting
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

        # Ellipse: circle -> scale by radii -> rotate by eigenvectors -> shift by mean
        ell = (circle * radii) @ evecs.T + mu
        ax.fill(ell[:, 0], ell[:, 1], color=color, alpha=alpha, linewidth=0.0)
        ax.plot(ell[:, 0], ell[:, 1], color=color, linewidth=2.0)

        # Center
        ax.scatter(mu[0], mu[1], color=color, s=40, zorder=3)

        # vMF arrow
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
    scale_by_layer2_weight=True,
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
    ax.set_title("Two-layer MoM parameters" if title is None else str(title))
    _set_axes_equal_3d(ax)
    return fig, ax


def plot_mom_components_2d(
    mom_or_layers,
    *,
    n_std=1.5,
    arrow_scale=1.0,
    normalize_kappa=True,
    scale_by_layer2_weight=True,
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

    # Validate 2D setting for layer 1 (Gaussian) and layer 2 (vMF)
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

        # Ellipse: circle -> scale by radii -> rotate by eigenvectors -> shift by mean
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
            #{i}:{j} (w2={w2:.2f})
            if annotate_arrows:
                tip = mean + vec
                ax.text(tip[0], tip[1], label, color=color)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D two-layer MoM parameters" if title is None else str(title))
    ax.grid(True, alpha=0.2)
    _set_axes_equal_2d(ax)
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

def mom_builder(dim: int, n_layer1_components, n_layer2_components, init_layer1, init_layer2, rng):
    layer1_mixture = clvm.MixtureModel(
        [clvm.MultivariateGaussian(dim) for _ in range(n_layer1_components)],
        init=init_layer1,
        rng=rng,
    )
    layer2_mixtures = [
        clvm.MixtureModel(
            [clvm.VonMisesFisher(dim) for _ in range(n_layer2_components)],
            init=init_layer2,
            rng=rng,
        )
        for _ in range(n_layer1_components)
    ]
    return clvm.TwoLayerMoM(layer1_mixture=layer1_mixture, layer2_mixtures=layer2_mixtures)

def mom_iso_builder(dim: int, n_layer1_components, n_layer2_components, init_layer1, init_layer2, rng):
    layer1_mixture = clvm.MixtureModel(
        [clvm.MultivariateGaussian(dim) for _ in range(n_layer1_components)],
        init=init_layer1,
        rng=rng,
    )
    layer2_mixtures = [
        clvm.MixtureModel(
            [clvm.VonMisesFisher(dim) for _ in range(n_layer2_components)],
            init=init_layer2,
            rng=rng,
        )
        for _ in range(n_layer1_components)
    ]
    return clvm.IsolatedTwoLayerMoM(layer1_mixture=layer1_mixture, layer2_mixtures=layer2_mixtures)


def cylindrical_mixture_builder(dim: int, n_components, init, rng):
    """Builder for 3D cylindrical mixtures used by BIC grid-search calibration."""
    return clvm.MixtureModel(
        [clvm.Cylindrical(d_gauss=dim, d_vmf=dim) for _ in range(n_components)],
        init=init,
        rng=rng,
    )

def ind_cylindrical_mixture_builder(dim: int, n_components, init, rng):
    """Builder for 3D cylindrical mixtures used by BIC grid-search calibration."""
    return clvm.MixtureModel(
        [clvm.IndCylindrical(d_gauss=dim, d_vmf=dim) for _ in range(n_components)],
        init=init,
        rng=rng,
    )

def train_all_models(dim: int,
                     x,
                     x_noisy,
                     setup: dict):

    x_gauss = x[:, :dim]
    x_vmf = x[:, dim:]
    x_noisy_gauss = x_noisy[:,:dim]
    x_noisy_vmf = x_noisy[:, dim:]

    print(f"EM iterations")

    print(f"Cylindrical mixture")
    cylmix = cylindrical_mixture_builder(
            dim = dim,
            n_components=setup["cylmix_l1"],
            init="k-means",
            rng=np.random.RandomState(42)
                ).fit(x)
    print(f"   01 No noise: {cylmix.n_iter}")
    noisy_cylmix = cylindrical_mixture_builder(
            dim=dim,
            n_components=setup["noisy_cylmix_l1"],
            init="k-means",
            rng=np.random.RandomState(42)
                ).fit(x_noisy)
    print(f"   02 Noisy: {noisy_cylmix.n_iter}")

    print(f"Independent cylindrical mixture")
    indcylmix = ind_cylindrical_mixture_builder(
            dim=dim,
            n_components=setup["indcylmix_l1"],
            init="k-means",
            rng=np.random.RandomState(42)
                ).fit(x)
    print(f"   03 No noise: {indcylmix.n_iter}")
    noisy_indcylmix = ind_cylindrical_mixture_builder(
            dim=dim,
            n_components=setup["noisy_indcylmix_l1"],
            init="k-means",
            rng=np.random.RandomState(42)
                ).fit(x_noisy)
    print(f"   04 Noisy: {noisy_indcylmix.n_iter}")

    print(f"Mixture of mixtures")
    mom = mom_builder(
            dim=dim,
            n_layer1_components=setup["mom_l1"],
            n_layer2_components=setup["mom_l2"],
            init_layer1="k-means",
            init_layer2="k-means",
            rng=np.random.RandomState(42)
                ).fit(x_gauss, x_vmf)
    print(f"   05 No noise: {mom.n_iter}")
    noisy_mom = mom_builder(
            dim=dim,
            n_layer1_components=setup["noisy_mom_l1"],
            n_layer2_components=setup["noisy_mom_l2"],
            init_layer1="k-means",
            init_layer2="k-means",
            rng=np.random.RandomState(42)
                ).fit(x_noisy_gauss, x_noisy_vmf)
    print(f"   06 Noisy: {noisy_mom.n_iter}")

    print("Isolation mixture of mixtures")
    isomom = mom_iso_builder(
            dim=dim,
            n_layer1_components=setup["isomom_l1"],
            n_layer2_components=setup["isomom_l2"],
            init_layer1="k-means",
            init_layer2="k-means",
            rng=np.random.RandomState(42)
                ).fit(x_gauss, x_vmf)
    print(f"   07 No noise: {isomom.n_iter}")
    noisy_isomom = mom_iso_builder(
            dim=dim,
            n_layer1_components=setup["noisy_isomom_l1"],
            n_layer2_components=setup["noisy_isomom_l2"],
            init_layer1="k-means",
            init_layer2="k-means",
            rng=np.random.RandomState(42)
                ).fit(x_noisy_gauss, x_noisy_vmf)
    print(f"   08 Noisy: {noisy_isomom.n_iter}")

    return [cylmix, noisy_cylmix, indcylmix, noisy_indcylmix, mom, noisy_mom, isomom, noisy_isomom]
