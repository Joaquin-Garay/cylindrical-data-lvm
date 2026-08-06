"""Plotting helpers for synthetic data experiments."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cyl_lvm.mixtures import MixtureModel

from .common import (
    _extract_cylindrical_gaussian_params,
    _is_mom_like,
    _safe_weights,
    unit,
)


CYLMIX_COMPARISON_METRICS = (
    "cond_cov",
    "cross_cov",
    "mu_gauss",
    "mu_vmf",
    "kappa_vmf",
)


def cylmix_comparison_dataframe(
    comparisons,
    *,
    label=None,
    metrics=CYLMIX_COMPARISON_METRICS,
):
    """
    Convert one or more ``cylmix_comparison`` results into a dataframe.

    ``comparisons`` can be either:
    - one comparison dictionary returned by ``clvm.cylmix_comparison``
    - a mapping from result labels to comparison dictionaries
    """
    metrics = tuple(metrics)
    if not metrics:
        raise ValueError("metrics must contain at least one metric name.")

    if _is_cylmix_comparison_result(comparisons, metrics):
        comparison_items = [(label or "comparison", comparisons)]
    elif isinstance(comparisons, Mapping):
        comparison_items = list(comparisons.items())
    else:
        raise TypeError(
            "comparisons must be a cylmix_comparison result or a mapping "
            "from labels to cylmix_comparison results."
        )

    rows = []
    for comparison_label, comparison in comparison_items:
        _validate_cylmix_comparison_result(comparison, metrics, name=str(comparison_label))
        matching = list(comparison["matching"])

        for row_idx, _ in enumerate(matching):
            row = {
                "comparison": str(comparison_label),
            }
            for metric in metrics:
                row[metric] = float(comparison[metric][row_idx])
            rows.append(row)

    return pd.DataFrame(
        rows,
        columns=["comparison", *metrics],
    )


def plot_cylmix_comparison_metrics(
    comparison_df,
    *,
    metrics=CYLMIX_COMPARISON_METRICS,
    max_cols=3,
    figsize=None,
    sharey=False,
    show=True,
    return_handles=False,
):
    """
    Plot per-component cylindrical-mixture comparison metrics.

    Pass the dataframe returned by ``cylmix_comparison_dataframe``. A raw
    ``cylmix_comparison`` result, or a mapping of labels to results, is also
    accepted and converted internally.
    """
    if not isinstance(comparison_df, pd.DataFrame):
        comparison_df = cylmix_comparison_dataframe(comparison_df, metrics=metrics)

    metrics = tuple(metrics)
    _validate_cylmix_comparison_dataframe(comparison_df, metrics)
    if not isinstance(max_cols, int) or max_cols < 1:
        raise ValueError("max_cols must be an integer >= 1.")

    plot_df = comparison_df.copy()
    plot_df["_component"] = plot_df.groupby("comparison", sort=False).cumcount()

    n_metrics = len(metrics)
    n_cols = min(max_cols, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    if figsize is None:
        figsize = (4.0 * n_cols, 3.0 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        sharey=sharey,
        constrained_layout=True,
    )

    for ax_idx, metric in enumerate(metrics):
        ax = axes.ravel()[ax_idx]
        metric_df = plot_df.pivot_table(
            index="_component",
            columns="comparison",
            values=metric,
            aggfunc="first",
            sort=False,
        )
        metric_df.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(metric)
        ax.set_xlabel("Component")
        ax.set_ylabel("Difference")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.2)

    for ax in axes.ravel()[n_metrics:]:
        ax.axis("off")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    for ax in axes.ravel():
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncols=len(labels),
        )
        layout_engine = fig.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(rect=(0.0, 0.0, 1.0, 0.92))

    if show:
        plt.show()

    if return_handles:
        return fig, axes

    plt.close(fig)
    return None


def _is_cylmix_comparison_result(value, metrics):
    return (
        isinstance(value, Mapping)
        and "matching" in value
        and all(metric in value for metric in metrics)
    )


def _validate_cylmix_comparison_result(comparison, metrics, *, name):
    if not _is_cylmix_comparison_result(comparison, metrics):
        raise ValueError(f"{name} must be a cylmix_comparison result.")

    matching = list(comparison["matching"])
    n_rows = len(matching)
    for metric in metrics:
        values = comparison[metric]
        if len(values) != n_rows:
            raise ValueError(
                f"{name}[{metric!r}] must have the same length as "
                f"{name}['matching']; got {len(values)} and {n_rows}."
            )


def _validate_cylmix_comparison_dataframe(comparison_df, metrics):
    required_columns = {"comparison", *metrics}
    missing_columns = sorted(required_columns.difference(comparison_df.columns))
    if missing_columns:
        raise ValueError(
            "comparison_df is missing required columns: "
            + ", ".join(missing_columns)
        )
    if comparison_df.empty:
        raise ValueError("comparison_df must contain at least one row.")


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


def _sync_3d_axes(axes):
    axes = np.asarray(axes).ravel()
    xlim = np.array([ax.get_xlim3d() for ax in axes], dtype=float)
    ylim = np.array([ax.get_ylim3d() for ax in axes], dtype=float)
    zlim = np.array([ax.get_zlim3d() for ax in axes], dtype=float)
    center = np.array([xlim.mean(), ylim.mean(), zlim.mean()])
    radius = 0.5 * max(
        xlim[:, 1].max() - xlim[:, 0].min(),
        ylim[:, 1].max() - ylim[:, 0].min(),
        zlim[:, 1].max() - zlim[:, 0].min(),
    )

    for ax in axes:
        ax.set_xlim3d(center[0] - radius, center[0] + radius)
        ax.set_ylim3d(center[1] - radius, center[1] + radius)
        ax.set_zlim3d(center[2] - radius, center[2] + radius)


def _plot_cross_corr_matrix_grid(
    mats,
    *,
    fig=None,
    axes=None,
    figsize=None,
    cmap="coolwarm",
    max_cols=4,
    title_prefix="Matrix",
):
    if len(mats) == 0:
        raise ValueError("No matrices to plot.")
    if not isinstance(max_cols, int) or max_cols < 1:
        raise ValueError("max_cols must be an integer >= 1.")

    n = len(mats)
    n_cols = min(max_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    if axes is None:
        if figsize is None:
            figsize = (3 * n_cols, 2 * n_rows)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
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
        ax.set_title(str(title_prefix) if n == 1 else f"{title_prefix} {k}")
        ax.set_xlabel("vMF dim")
        ax.set_ylabel("Gaussian dim")
        ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(M.shape[0]))

    for k in range(n, n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row, col].axis("off")

    return fig, axes, im


def _resolve_full_cov_split(full_cov, d_gauss, *, name):
    full_cov = np.asarray(full_cov, dtype=float)
    if full_cov.ndim != 2 or full_cov.shape[0] != full_cov.shape[1]:
        raise ValueError(f"{name} must be a square 2D covariance matrix.")

    n_features = full_cov.shape[0]
    if d_gauss is None:
        if n_features % 2 != 0:
            raise ValueError(
                f"{name} has odd size {n_features}; provide d_gauss explicitly."
            )
        d_gauss = n_features // 2

    if not isinstance(d_gauss, (int, np.integer)):
        raise ValueError("d_gauss must be an integer.")
    d_gauss = int(d_gauss)
    if d_gauss <= 0 or d_gauss >= n_features:
        raise ValueError(
            f"d_gauss must be between 1 and {n_features - 1}; got {d_gauss}."
        )

    return full_cov, d_gauss


def _cross_corr_from_full_cov(full_cov, *, d_gauss=None, name="full_cov"):
    full_cov, d_gauss = _resolve_full_cov_split(full_cov, d_gauss, name=name)
    cov_gg = full_cov[:d_gauss, :d_gauss]
    cov_vv = full_cov[d_gauss:, d_gauss:]
    cross_cov = full_cov[:d_gauss, d_gauss:]

    gauss_var = np.diag(cov_gg)
    vmf_var = np.diag(cov_vv)
    if np.any(~np.isfinite(gauss_var)) or np.any(gauss_var <= 0.0):
        raise ValueError(f"{name} first block must have positive finite variances.")
    if np.any(~np.isfinite(vmf_var)) or np.any(vmf_var <= 0.0):
        raise ValueError(f"{name} second block must have positive finite variances.")

    return cross_cov / np.sqrt(gauss_var)[:, None] / np.sqrt(vmf_var)[None, :]


def _cross_corr_matrices_from_full_covs(full_covs, *, d_gauss=None, name="full_cov"):
    try:
        full_covs = np.asarray(full_covs, dtype=float)
    except ValueError:
        if not isinstance(full_covs, (list, tuple)):
            raise
        matrices = [np.asarray(cov, dtype=float) for cov in full_covs]
    else:
        if full_covs.ndim == 2:
            matrices = [full_covs]
        elif full_covs.ndim == 3:
            matrices = [full_covs[i] for i in range(full_covs.shape[0])]
        else:
            raise ValueError(
                f"{name} must be a full covariance matrix or a stack of matrices."
            )

    if not matrices:
        raise ValueError(f"No {name} matrices to plot.")

    return [
        _cross_corr_from_full_cov(
            cov,
            d_gauss=d_gauss,
            name=name if len(matrices) == 1 else f"{name}[{i}]",
        )
        for i, cov in enumerate(matrices)
    ]


def _mixture_weights(mixture, *, name):
    if hasattr(mixture, "layer1_mixture") and hasattr(mixture, "layer2_mixtures"):
        layer1_mixture = mixture.layer1_mixture
        layer2_mixtures = list(mixture.layer2_mixtures)
        if len(layer2_mixtures) != len(layer1_mixture.components):
            raise ValueError(
                f"{name} layer-2 mixture count must match layer-1 components."
            )
        if not all(
            getattr(layer2_mixture, "n_components", None) == 1
            for layer2_mixture in layer2_mixtures
        ):
            raise ValueError(
                f"{name} must be a MixtureModel instance or a TwoLayerMoM "
                "with exactly one layer-2 component per layer-1 component."
            )
        return _mixture_weights(layer1_mixture, name=f"{name}.layer1_mixture")

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


def _validate_component_dimension(dimension):
    if not isinstance(dimension, (int, np.integer)) or int(dimension) not in (2, 3):
        raise ValueError("dimension must be either 2 or 3.")
    return int(dimension)


def _normalize_model_kind(model, model_kind):
    if model_kind == "auto":
        return "mom" if _is_mom_like(model) else "cylindrical"

    if not isinstance(model_kind, str):
        raise ValueError("model_kind must be one of {'auto', 'cylindrical', 'mom'}.")

    normalized = model_kind.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cylindrical": "cylindrical",
        "cylindrical_mixture": "cylindrical",
        "cylmix": "cylindrical",
        "mom": "mom",
        "mixture_of_mixtures": "mom",
        "two_layer_mom": "mom",
    }
    if normalized not in aliases:
        raise ValueError("model_kind must be one of {'auto', 'cylindrical', 'mom'}.")
    return aliases[normalized]


def _component_list(value, *, name):
    components = list(value.components) if hasattr(value, "components") else list(value)
    if not components:
        raise ValueError(f"No {name} to plot.")
    return components


def _mom_layers(mom_or_layers):
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
    return layer1_components, layer2_mixtures


def _component_shape_template(dimension, n_shape):
    if dimension == 2:
        theta = np.linspace(0.0, 2.0 * np.pi, n_shape)
        return np.column_stack((np.cos(theta), np.sin(theta)))

    u = np.linspace(0.0, 2.0 * np.pi, n_shape)
    v = np.linspace(0.0, np.pi, n_shape)
    uu, vv = np.meshgrid(u, v)
    return np.stack(
        [np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)],
        axis=-1,
    )


def _make_component_axes(dimension, *, figsize, ax):
    if ax is not None:
        return ax.figure, ax

    if dimension == 2:
        return plt.subplots(figsize=figsize)

    fig = plt.figure(figsize=figsize)
    return fig, fig.add_subplot(111, projection="3d")


def _plot_gaussian_region(ax, dimension, template, mean, cov, *, n_std, color, alpha):
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    radii = n_std * np.sqrt(evals)
    region = (template * radii) @ evecs.T + mean

    if dimension == 2:
        ax.fill(region[:, 0], region[:, 1], color=color, alpha=alpha, linewidth=0.0)
        ax.plot(region[:, 0], region[:, 1], color=color, linewidth=2.0)
        ax.scatter(mean[0], mean[1], color=color, s=40, zorder=3)
        return

    ax.plot_surface(
        region[..., 0], region[..., 1], region[..., 2],
        rstride=1, cstride=1, linewidth=0.0, antialiased=True,
        alpha=alpha, color=color
    )
    ax.scatter(mean[0], mean[1], mean[2], color=color, s=40)


def _plot_direction_arrow(ax, dimension, mean, vec, *, color, linewidth, alpha):
    if dimension == 2:
        ax.quiver(
            mean[0], mean[1],
            vec[0], vec[1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            width=0.006,
            zorder=4,
        )
        return

    ax.quiver(
        mean[0], mean[1], mean[2],
        vec[0], vec[1], vec[2],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        arrow_length_ratio=0.15,
    )


def plot_cylindrical_sample_2d(
    x,
    *,
    labels=None,
    figsize=(6, 6),
    arrow_scale=0.15,
    point_size=12,
    point_alpha=0.55,
    arrow_alpha=0.85,
    max_points=None,
    random_state=42,
    normalize_directions=True,
    legend=True,
    ax=None,
    title=None,
):
    """Plot a 2D cylindrical sample as points with directional arrows.

    Expects ``x`` with shape ``(N, 4)``. The first two columns are plotted as
    Euclidean coordinates. The last two columns give the arrow direction for
    each observation.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError("x must have shape (N, 4).")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values.")

    n_obs = x.shape[0]
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.shape[0] != n_obs:
            raise ValueError("labels must have shape (N,).")

    if max_points is not None:
        if not isinstance(max_points, (int, np.integer)) or int(max_points) < 1:
            raise ValueError("max_points must be an integer >= 1.")
        max_points = int(max_points)
        if n_obs > max_points:
            if isinstance(random_state, np.random.RandomState):
                rng = random_state
            else:
                rng = np.random.RandomState(random_state)
            idx = np.sort(rng.choice(n_obs, size=max_points, replace=False))
            x = x[idx]
            if labels is not None:
                labels = labels[idx]

    x_gauss = x[:, :2]
    directions = x[:, 2:]
    if normalize_directions:
        directions = unit(directions)

    vectors = arrow_scale * directions

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cmap = plt.cm.tab10
    if labels is None:
        color = cmap(0)
        ax.scatter(
            x_gauss[:, 0],
            x_gauss[:, 1],
            s=point_size,
            alpha=point_alpha,
            color=color,
            linewidths=0.0,
            zorder=2,
        )
        ax.quiver(
            x_gauss[:, 0],
            x_gauss[:, 1],
            vectors[:, 0],
            vectors[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            alpha=arrow_alpha,
            width=0.0035,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
            zorder=3,
        )
    else:
        unique_labels = np.unique(labels)
        for i, value in enumerate(unique_labels):
            mask = labels == value
            color = cmap(i % 10)
            ax.scatter(
                x_gauss[mask, 0],
                x_gauss[mask, 1],
                s=point_size,
                alpha=point_alpha,
                color=color,
                linewidths=0.0,
                label=str(value),
                zorder=2,
            )
            ax.quiver(
                x_gauss[mask, 0],
                x_gauss[mask, 1],
                vectors[mask, 0],
                vectors[mask, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                alpha=arrow_alpha,
                width=0.0035,
                headwidth=3.5,
                headlength=4.5,
                headaxislength=4.0,
                zorder=3,
            )
        if legend:
            ax.legend(title="label", frameon=False)

    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
    ax.set_title("2D cylindrical sample" if title is None else str(title))
    ax.grid(True, alpha=0.2)
    _set_axes_equal_2d(ax)
    return fig, ax


def _validate_axis_labels(labels, *, name, size):
    labels = tuple(labels)
    if len(labels) != size:
        raise ValueError(f"{name} must contain exactly {size} labels.")
    return tuple(str(label) for label in labels)


def _sample_rows(x, labels, *, max_points, random_state):
    n_obs = x.shape[0]
    if max_points is None:
        return x, labels

    if not isinstance(max_points, (int, np.integer)) or int(max_points) < 1:
        raise ValueError("max_points must be an integer >= 1.")
    max_points = int(max_points)
    if n_obs <= max_points:
        return x, labels

    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)
    idx = np.sort(rng.choice(n_obs, size=max_points, replace=False))
    x = x[idx]
    if labels is not None:
        labels = labels[idx]
    return x, labels


def _make_sample_3d_circle_axes(*, figsize, axes):
    if axes is None:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax_space = fig.add_subplot(1, 2, 1, projection="3d")
        ax_circle = fig.add_subplot(1, 2, 2)
        return fig, ax_space, ax_circle

    axes_arr = np.asarray(axes, dtype=object).ravel()
    if axes_arr.size != 2:
        raise ValueError("axes must contain exactly two axes.")
    ax_space, ax_circle = axes_arr
    fig = ax_space.figure
    if ax_circle.figure is not fig:
        raise ValueError("Both axes must belong to the same figure.")
    return fig, ax_space, ax_circle


def _draw_unit_circle(ax, *, labels):
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(np.cos(theta), np.sin(theta), color="0.45", linewidth=1.0)
    ax.axhline(0.0, color="0.85", linewidth=0.8)
    ax.axvline(0.0, color="0.85", linewidth=0.8)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


def plot_cylindrical_sample_3d(
    x,
    *,
    labels=None,
    figsize=(11.0, 5.0),
    point_size=16,
    point_alpha=0.62,
    max_points=None,
    random_state=42,
    normalize_directions=True,
    linear_feature_names=("x", "y", "z"),
    direction_feature_names=("cos(theta)", "sin(theta)"),
    legend=True,
    axes=None,
    title="3D cylindrical sample",
):
    """
    Plot a 3D cylindrical sample as Euclidean points plus unit-circle points.

    Expects ``x`` with shape ``(N, 5)``. The first three columns are plotted in
    3D Euclidean space. The last two columns are plotted on the unit circle.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("x must have shape (N, 5).")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values.")

    n_obs = x.shape[0]
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.shape[0] != n_obs:
            raise ValueError("labels must have shape (N,).")

    x, labels = _sample_rows(
        x,
        labels,
        max_points=max_points,
        random_state=random_state,
    )

    x_gauss = x[:, :3]
    directions = x[:, 3:]
    if normalize_directions:
        directions = unit(directions)

    linear_feature_names = _validate_axis_labels(
        linear_feature_names,
        name="linear_feature_names",
        size=3,
    )
    direction_feature_names = _validate_axis_labels(
        direction_feature_names,
        name="direction_feature_names",
        size=2,
    )

    fig, ax_space, ax_circle = _make_sample_3d_circle_axes(
        figsize=figsize,
        axes=axes,
    )
    _draw_unit_circle(ax_circle, labels=direction_feature_names)

    cmap = plt.cm.tab10
    if labels is None:
        color = cmap(0)
        ax_space.scatter(
            x_gauss[:, 0],
            x_gauss[:, 1],
            x_gauss[:, 2],
            s=point_size,
            alpha=point_alpha,
            color=color,
            linewidths=0.0,
        )
        ax_circle.scatter(
            directions[:, 0],
            directions[:, 1],
            s=point_size,
            alpha=point_alpha,
            color=color,
            linewidths=0.0,
            zorder=3,
        )
    else:
        unique_labels = np.unique(labels)
        for i, value in enumerate(unique_labels):
            mask = labels == value
            color = cmap(i % 10)
            label = str(value)
            ax_space.scatter(
                x_gauss[mask, 0],
                x_gauss[mask, 1],
                x_gauss[mask, 2],
                s=point_size,
                alpha=point_alpha,
                color=color,
                linewidths=0.0,
                label=label,
            )
            ax_circle.scatter(
                directions[mask, 0],
                directions[mask, 1],
                s=point_size,
                alpha=point_alpha,
                color=color,
                linewidths=0.0,
                label=label,
                zorder=3,
            )

        if legend:
            ax_circle.legend(title="label", frameon=False, loc="upper right")

    ax_space.set_xlabel(linear_feature_names[0])
    ax_space.set_ylabel(linear_feature_names[1])
    ax_space.set_zlabel(linear_feature_names[2])
    ax_space.set_title("3D Euclidean points")
    _set_axes_equal_3d(ax_space)

    ax_circle.set_title("Circular points")
    if title is not None:
        fig.suptitle(str(title))

    return fig, np.asarray([ax_space, ax_circle], dtype=object)


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


def plot_components(
    dimension,
    model,
    *,
    model_kind="auto",
    figsize=None,
    n_std=1.5,
    arrow_scale=1.0,
    normalize_kappa=True,
    alpha=0.22,
    n_ellipse=200,
    n_ellipsoid=36,
    scale_by_layer2_weight=False,
    ax=None,
    title=None,
):
    """Plot 2D/3D components for a cylindrical mixture or TwoLayerMoM."""
    dimension = _validate_component_dimension(dimension)
    model_kind = _normalize_model_kind(model, model_kind)
    if figsize is None:
        figsize = (5, 5) if dimension == 2 else (8, 7)

    plot_items = []
    kappas = []
    if model_kind == "mom":
        layer1_components, layer2_mixtures = _mom_layers(model)
        for i, (gauss, vmf_mix) in enumerate(zip(layer1_components, layer2_mixtures)):
            mean, cov = (np.asarray(v, dtype=float) for v in gauss.params)
            if mean.shape != (dimension,) or cov.shape != (dimension, dimension):
                raise ValueError(
                    f"This plotter expects layer-1 Gaussian components in {dimension}D."
                )

            vmfs = _component_list(
                vmf_mix,
                name=f"layer2_mixtures[{i}].components",
            )
            weights = _safe_weights(vmf_mix)
            if weights is None or weights.size != len(vmfs):
                weights = np.full(len(vmfs), 1.0 / len(vmfs))

            arrows = []
            for vmf, weight in zip(vmfs, weights):
                if getattr(vmf, "d", None) != dimension:
                    raise ValueError(
                        f"This plotter expects layer-2 vMF components in {dimension}D."
                    )
                kappa = float(vmf.kappa)
                arrows.append((np.asarray(vmf.mu, dtype=float), kappa, float(weight)))
                kappas.append(kappa)

            plot_items.append((mean, cov, arrows))
        default_title = (
            "2D two-layer MoM parameters"
            if dimension == 2
            else "Two-layer MoM parameters"
        )
    else:
        components = _component_list(model, name="components")
        for i, comp in enumerate(components):
            if (
                getattr(comp, "d_gauss", None) != dimension
                or getattr(comp, "d_vmf", None) != dimension
            ):
                raise ValueError(
                    f"This plotter expects d_gauss={dimension} and d_vmf={dimension} "
                    "for all components."
                )
            if not hasattr(comp, "vmf"):
                raise ValueError(f"components[{i}] must expose a vmf component.")

            mean, cov = _extract_cylindrical_gaussian_params(comp)
            kappa = float(comp.vmf.kappa)
            plot_items.append(
                (
                    np.asarray(mean, dtype=float),
                    np.asarray(cov, dtype=float),
                    [(np.asarray(comp.vmf.mu, dtype=float), kappa, 1.0)],
                )
            )
            kappas.append(kappa)

        default_title = (
            "2D cylindrical mixture parameters"
            if dimension == 2
            else "Cylindrical mixture parameters"
        )
        scale_by_layer2_weight = False

    kappas = np.asarray(kappas, dtype=float)
    if kappas.size == 0:
        raise ValueError("No vMF arrows to plot.")
    kappa_ref = np.max(kappas) if normalize_kappa else 1.0
    if not np.isfinite(kappa_ref) or kappa_ref <= 0.0:
        kappa_ref = 1.0

    n_shape = n_ellipse if dimension == 2 else n_ellipsoid
    fig, ax = _make_component_axes(dimension, figsize=figsize, ax=ax)
    template = _component_shape_template(dimension, n_shape)
    cmap = plt.cm.tab10

    for i, (mean, cov, arrows) in enumerate(plot_items):
        color = cmap(i % 10)
        _plot_gaussian_region(
            ax,
            dimension,
            template,
            mean,
            cov,
            n_std=n_std,
            color=color,
            alpha=alpha,
        )

        for direction, kappa, weight in arrows:
            length = arrow_scale * (kappa / kappa_ref)
            if scale_by_layer2_weight:
                length *= weight

            _plot_direction_arrow(
                ax,
                dimension,
                mean,
                length * direction,
                color=color,
                linewidth=1.0 + 1.5 * weight,
                alpha=0.45 + 0.5 * weight,
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(default_title if title is None else str(title))
    if dimension == 2:
        ax.grid(True, alpha=0.2)
        _set_axes_equal_2d(ax)
    else:
        ax.set_zlabel("z")
        _set_axes_equal_3d(ax)

    return fig, ax


def plot_model_vs_generator(
    dimension,
    model,
    generator,
    *,
    title=None,
    model_title=None,
    generator_title="Generator",
    model_kind="auto",
    generator_kind="auto",
    arrow_scale=None,
    model_arrow_scale=None,
    generator_arrow_scale=None,
    normalize_kappa=False,
    n_std=1.5,
    alpha=0.22,
    n_ellipse=200,
    n_ellipsoid=36,
    scale_by_layer2_weight=False,
    share_limits=True,
    figsize=None,
):
    """
    Plot model and generator components side by side.

    Both inputs must resolve to the same plotting class: either cylindrical
    mixture-like or TwoLayerMoM-like.
    """
    dimension = _validate_component_dimension(dimension)
    model_kind = _normalize_model_kind(model, model_kind)
    generator_kind = _normalize_model_kind(generator, generator_kind)
    if model_kind != generator_kind:
        raise ValueError(
            "model and generator must have the same plotting class; "
            f"got {model_kind!r} and {generator_kind!r}."
        )

    if arrow_scale is not None and model_arrow_scale is not None:
        raise ValueError("Provide only one of arrow_scale or model_arrow_scale.")
    if arrow_scale is not None and generator_arrow_scale is not None:
        raise ValueError("Provide only one of arrow_scale or generator_arrow_scale.")

    if model_arrow_scale is None:
        model_arrow_scale = arrow_scale
    if generator_arrow_scale is None:
        generator_arrow_scale = arrow_scale
    if model_arrow_scale is None:
        model_arrow_scale = 0.1 if model_kind == "mom" else 0.3
    if generator_arrow_scale is None:
        generator_arrow_scale = model_arrow_scale
    if model_title is None:
        model_title = "Model" if title is None else str(title)
    if figsize is None:
        figsize = (10, 5) if dimension == 2 else (14, 7)

    if dimension == 2:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
        )
        left_ax, right_ax = axes[0]
    else:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        left_ax = fig.add_subplot(121, projection="3d")
        right_ax = fig.add_subplot(122, projection="3d")
        axes = np.asarray([left_ax, right_ax])

    plot_components(
        dimension,
        model,
        n_std=n_std,
        arrow_scale=model_arrow_scale,
        normalize_kappa=normalize_kappa,
        scale_by_layer2_weight=scale_by_layer2_weight,
        alpha=alpha,
        n_ellipse=n_ellipse,
        n_ellipsoid=n_ellipsoid,
        model_kind=model_kind,
        ax=left_ax,
        title=model_title,
    )

    plot_components(
        dimension,
        generator,
        model_kind=generator_kind,
        n_std=n_std,
        arrow_scale=generator_arrow_scale,
        normalize_kappa=normalize_kappa,
        scale_by_layer2_weight=scale_by_layer2_weight,
        alpha=alpha,
        n_ellipse=n_ellipse,
        n_ellipsoid=n_ellipsoid,
        ax=right_ax,
        title=generator_title,
    )

    if share_limits:
        if dimension == 2:
            _sync_2d_axes([left_ax, right_ax])
        else:
            _sync_3d_axes([left_ax, right_ax])

    return fig, axes.ravel()


def plot_cross_corr_matrices(
    full_cov,
    *,
    d_gauss=None,
    figsize=None,
    cmap="coolwarm",
    max_cols=4,
    title_prefix="Cross correlation",
    show=True,
    return_handles=False,
):
    """Plot cross-correlation matrices from full covariance matrices."""
    mats = _cross_corr_matrices_from_full_covs(full_cov, d_gauss=d_gauss)
    fig, axes, im = _plot_cross_corr_matrix_grid(
        mats,
        figsize=figsize,
        cmap=cmap,
        max_cols=max_cols,
        title_prefix=title_prefix,
    )

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)

    if show:
        plt.show()

    if return_handles:
        return fig, axes

    plt.close(fig)
    return None


def plot_cross_corr_comparison(
    left_full_cov,
    right_full_cov,
    *,
    d_gauss=None,
    left_d_gauss=None,
    right_d_gauss=None,
    left_title="Left",
    right_title="Right",
    figsize=None,
    cmap="coolwarm",
    max_cols=None,
    show=True,
    return_handles=False,
):
    """Plot cross-correlation matrices from two full covariance inputs."""
    if left_d_gauss is None:
        left_d_gauss = d_gauss
    if right_d_gauss is None:
        right_d_gauss = d_gauss

    left_mats = _cross_corr_matrices_from_full_covs(
        left_full_cov,
        d_gauss=left_d_gauss,
        name="left_full_cov",
    )
    right_mats = _cross_corr_matrices_from_full_covs(
        right_full_cov,
        d_gauss=right_d_gauss,
        name="right_full_cov",
    )

    max_count = max(len(left_mats), len(right_mats))
    if max_cols is None:
        n_cols = max_count
    else:
        if not isinstance(max_cols, int) or max_cols < 1:
            raise ValueError("max_cols must be an integer >= 1 or None.")
        n_cols = min(max_cols, max_count)
    n_rows = int(np.ceil(max_count / n_cols))
    if figsize is None:
        figsize = (3 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(
        2 * n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    left_axes = axes[:n_rows, :]
    right_axes = axes[n_rows:, :]

    _, left_axes, left_im = _plot_cross_corr_matrix_grid(
        left_mats,
        fig=fig,
        axes=left_axes,
        cmap=cmap,
        max_cols=n_cols,
        title_prefix=left_title,
    )
    _, right_axes, right_im = _plot_cross_corr_matrix_grid(
        right_mats,
        fig=fig,
        axes=right_axes,
        cmap=cmap,
        max_cols=n_cols,
        title_prefix=right_title,
    )

    fig.colorbar(
        left_im if left_im is not None else right_im,
        ax=axes.ravel().tolist(),
        shrink=0.85,
    )

    if show:
        plt.show()

    if return_handles:
        return fig, axes

    plt.close(fig)
    return None
