"""Plotting helpers for ocean experiment grid-search results."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from cyl_lvm import MultivariateGaussian, VonMisesFisher


def _extract_config_results(search_result, *, name: str) -> list[dict[str, Any]]:
    if isinstance(search_result, Mapping):
        if "config_results" not in search_result:
            raise ValueError(f"{name} dictionary must contain a 'config_results' key.")
        rows = search_result["config_results"]
    else:
        rows = search_result

    rows = list(rows)
    if not rows:
        raise ValueError(f"{name} must contain at least one config-result row.")
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError(f"{name} rows must be dictionaries.")
    return [dict(row) for row in rows]


def _finite_successful_config_rows(
        rows: list[dict[str, Any]],
        *,
        required_keys: tuple[str, ...],
        metric: str,
        ) -> list[dict[str, Any]]:
    missing = [
        key
        for key in (*required_keys, metric)
        if any(key not in row for row in rows)
    ]
    if missing:
        unique_missing = ", ".join(sorted(set(missing)))
        raise ValueError(f"config_results are missing required columns: {unique_missing}.")

    out = []
    for row in rows:
        if not row.get("success", True):
            continue
        value = float(row[metric])
        if np.isfinite(value):
            out.append(row)
    if not out:
        raise ValueError(f"No successful finite {metric!r} values to plot.")
    return out


def _set_axes_equal_3d(ax) -> None:
    xlim = np.asarray(ax.get_xlim3d(), dtype=float)
    ylim = np.asarray(ax.get_ylim3d(), dtype=float)
    zlim = np.asarray(ax.get_zlim3d(), dtype=float)
    center = np.array([xlim.mean(), ylim.mean(), zlim.mean()])
    radius = 0.5 * max(
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        zlim[1] - zlim[0],
    )
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 1.0
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


def _validate_axis_labels(labels, *, name: str, size: int) -> tuple[str, ...]:
    labels = tuple(labels)
    if len(labels) != size:
        raise ValueError(f"{name} must contain exactly {size} labels.")
    return tuple(str(label) for label in labels)


def _validate_positive_float(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return value


def _validate_unit_interval(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


def _save_figure(fig, save_path, *, save_kwargs: Mapping[str, Any] | None = None) -> None:
    if save_path is None:
        return
    if isinstance(save_path, str) and not save_path.strip():
        raise ValueError("save_path must not be empty.")

    try:
        path = Path(save_path)
    except TypeError as exc:
        raise ValueError("save_path must be a path-like string.") from exc

    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "bbox_inches": "tight",
        "facecolor": fig.get_facecolor(),
        "edgecolor": fig.get_edgecolor(),
    }
    if save_kwargs is not None:
        if not isinstance(save_kwargs, Mapping):
            raise ValueError("save_kwargs must be a mapping.")
        kwargs.update(dict(save_kwargs))

    fig.savefig(path, **kwargs)


def _apply_light_background(fig) -> None:
    fig.patch.set_facecolor("white")
    if fig._suptitle is not None:
        fig._suptitle.set_color("black")

    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.title.set_color("black")

        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = getattr(ax, axis_name, None)
            if axis is None:
                continue
            axis.label.set_color("black")
            if hasattr(axis, "pane"):
                axis.pane.set_facecolor("white")
                axis.pane.set_edgecolor("0.75")
                axis.pane.set_alpha(1.0)
            if hasattr(axis, "_axinfo"):
                axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)

        for axis_name in ("x", "y", "z"):
            try:
                ax.tick_params(axis=axis_name, colors="black")
            except ValueError:
                pass

        for spine in ax.spines.values():
            spine.set_color("0.35")

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("black")
            legend.get_title().set_color("black")
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("0.75")


def _finalize_figure(
        fig,
        *,
        show: bool,
        save_path=None,
        save_kwargs: Mapping[str, Any] | None = None,
        light_background: bool = True,
        ) -> None:
    if light_background:
        _apply_light_background(fig)

    _save_figure(fig, save_path, save_kwargs=save_kwargs)

    if show:
        import matplotlib.pyplot as plt

        plt.show()


def _validate_cylindrical_covariance(covariance: str) -> str:
    if not isinstance(covariance, str):
        raise ValueError("covariance must be one of {'unconditional', 'conditional'}.")
    covariance = covariance.strip().lower()
    aliases = {
        "unconditional": "unconditional",
        "marginal": "unconditional",
        "conditional": "conditional",
        "cond": "conditional",
    }
    if covariance not in aliases:
        raise ValueError("covariance must be one of {'unconditional', 'conditional'}.")
    return aliases[covariance]


def _cylindrical_model(value, *, name: str):
    if isinstance(value, Mapping):
        if "best_model" not in value:
            raise ValueError(f"{name} dictionary must contain a 'best_model' key.")
        value = value["best_model"]

    if not hasattr(value, "components") or not hasattr(value, "n_components"):
        raise ValueError(f"{name} must be a cylindrical MixtureModel or grid-search result.")

    components = list(value.components)
    if not components:
        raise ValueError(f"{name} must contain at least one component.")
    if int(value.n_components) != len(components):
        raise ValueError(
            f"{name}.n_components must match the number of components; "
            f"got {value.n_components} and {len(components)}."
        )

    return value, components


def _two_layer_model(value, *, name: str):
    if isinstance(value, Mapping):
        if "best_model" not in value:
            raise ValueError(f"{name} dictionary must contain a 'best_model' key.")
        value = value["best_model"]

    if not hasattr(value, "layer1_mixture") or not hasattr(value, "layer2_mixtures"):
        raise ValueError(f"{name} must be a TwoLayerMoM-like model or grid-search result.")

    layer1_mixture = value.layer1_mixture
    layer2_mixtures = list(value.layer2_mixtures)
    if not hasattr(layer1_mixture, "components") or not hasattr(layer1_mixture, "n_components"):
        raise ValueError(f"{name}.layer1_mixture must be a MixtureModel-like object.")

    layer1_components = list(layer1_mixture.components)
    if not layer1_components:
        raise ValueError(f"{name}.layer1_mixture must contain at least one component.")
    if int(layer1_mixture.n_components) != len(layer1_components):
        raise ValueError(
            f"{name}.layer1_mixture.n_components must match the number of components; "
            f"got {layer1_mixture.n_components} and {len(layer1_components)}."
        )
    if len(layer2_mixtures) != len(layer1_components):
        raise ValueError(
            f"{name}.layer2_mixtures length must match layer-1 components; "
            f"got {len(layer2_mixtures)} and {len(layer1_components)}."
        )

    return value, layer1_mixture, layer1_components, layer2_mixtures


def _mixture_weights(model, *, n_components: int, name: str) -> np.ndarray:
    try:
        weights = np.asarray(model.weights, dtype=float)
    except Exception as exc:
        raise ValueError(f"{name} weights are not initialized.") from exc

    if weights.shape != (n_components,):
        raise ValueError(
            f"{name} weights must have shape ({n_components},), got {weights.shape}."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} weights contain non-finite values.")
    if np.any(weights < 0.0):
        raise ValueError(f"{name} weights must be non-negative.")

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} weights must sum to a positive finite value.")
    return weights / total


def _mixture_components(mixture, *, name: str) -> list[Any]:
    if not hasattr(mixture, "components") or not hasattr(mixture, "n_components"):
        raise ValueError(f"{name} must be a MixtureModel-like object.")
    components = list(mixture.components)
    if not components:
        raise ValueError(f"{name} must contain at least one component.")
    if int(mixture.n_components) != len(components):
        raise ValueError(
            f"{name}.n_components must match the number of components; "
            f"got {mixture.n_components} and {len(components)}."
        )
    return components


def _validate_vector(value, *, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _validate_covariance(value, *, size: int, name: str) -> np.ndarray:
    cov = np.asarray(value, dtype=float)
    if cov.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {cov.shape}.")
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{name} contains non-finite values.")
    cov = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    if np.min(eigvals) < -1e-8:
        raise ValueError(f"{name} must be positive semi-definite.")
    return cov


def _extract_gaussian_component_params(
        component,
        *,
        idx: int,
        size: int,
        name: str,
        ) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(component, "params"):
        raise ValueError(f"{name}[{idx}] must expose Gaussian params.")
    try:
        mean, cov = component.params
    except Exception as exc:
        raise ValueError(f"{name}[{idx}].params must unpack as (mean, covariance).") from exc
    return (
        _validate_vector(mean, size=size, name=f"{name}[{idx}].mean"),
        _validate_covariance(cov, size=size, name=f"{name}[{idx}].covariance"),
    )


def _extract_vmf_component_params(
        component,
        *,
        size: int,
        name: str,
        ) -> tuple[np.ndarray, float]:
    if not hasattr(component, "mu") or not hasattr(component, "kappa"):
        raise ValueError(f"{name} must expose vMF mu and kappa.")
    direction = _validate_vector(
        component.mu,
        size=size,
        name=f"{name}.mu",
    )
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0.0:
        raise ValueError(f"{name}.mu must be non-zero.")
    direction = direction / direction_norm

    kappa = float(component.kappa)
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError(f"{name}.kappa must be positive and finite.")
    return direction, kappa


def _component_family(component) -> str:
    if isinstance(component, VonMisesFisher):
        return "vmf"
    if isinstance(component, MultivariateGaussian):
        return "gaussian"

    return "unknown"


def _infer_two_layer_family(layer1_components, layer2_mixtures) -> tuple[str, str]:
    layer1_families = {_component_family(component) for component in layer1_components}
    if len(layer1_families) != 1:
        raise ValueError("All layer-1 components must belong to the same supported family.")
    layer1_family = next(iter(layer1_families))

    layer2_families = set()
    for layer1_idx, layer2_mixture in enumerate(layer2_mixtures):
        layer2_components = _mixture_components(
            layer2_mixture,
            name=f"layer2_mixtures[{layer1_idx}]",
        )
        layer2_families.update(_component_family(component) for component in layer2_components)

    if len(layer2_families) != 1:
        raise ValueError("All layer-2 components must belong to the same supported family.")
    layer2_family = next(iter(layer2_families))

    family = (layer1_family, layer2_family)
    if family not in {("gaussian", "vmf"), ("vmf", "gaussian")}:
        raise ValueError(
            "plot_two_layer_mixture_parameters supports only Gaussian->vMF "
            "and vMF->Gaussian two-layer models."
        )
    return family


def _extract_cylindrical_component_params(
        component,
        *,
        idx: int,
        covariance: str,
        ) -> dict[str, Any]:
    d_gauss = getattr(component, "d_gauss", None)
    d_vmf = getattr(component, "d_vmf", None)
    if d_gauss != 3 or d_vmf != 2:
        raise ValueError(
            "Cylindrical parameter plots expect each component to have "
            f"d_gauss=3 and d_vmf=2; component {idx} has "
            f"d_gauss={d_gauss!r}, d_vmf={d_vmf!r}."
        )

    mean = _validate_vector(
        component.mu_gauss,
        size=3,
        name=f"components[{idx}].mu_gauss",
    )
    if covariance == "unconditional":
        cov = _validate_covariance(
            component.unconditional_gauss_cov,
            size=3,
            name=f"components[{idx}].unconditional_gauss_cov",
        )
    else:
        cov = _validate_covariance(
            component.cond_cov,
            size=3,
            name=f"components[{idx}].cond_cov",
        )

    if not hasattr(component, "vmf"):
        raise ValueError(f"components[{idx}] must expose a vMF component.")
    vmf = component.vmf
    direction, kappa = _extract_vmf_component_params(
        vmf,
        size=2,
        name=f"components[{idx}].vmf",
    )

    return {
        "mean": mean,
        "cov": cov,
        "direction": direction,
        "kappa": kappa,
    }


def _ellipsoid_template(n_ellipsoid: int) -> np.ndarray:
    if not isinstance(n_ellipsoid, (int, np.integer)) or int(n_ellipsoid) < 8:
        raise ValueError("n_ellipsoid must be an integer >= 8.")
    n_ellipsoid = int(n_ellipsoid)
    u = np.linspace(0.0, 2.0 * np.pi, n_ellipsoid)
    v = np.linspace(0.0, np.pi, n_ellipsoid)
    uu, vv = np.meshgrid(u, v)
    return np.stack(
        [np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)],
        axis=-1,
    )


def _plot_gaussian_ellipsoid(
        ax,
        template: np.ndarray,
        mean: np.ndarray,
        cov: np.ndarray,
        *,
        n_std: float,
        color,
        alpha: float,
        ) -> None:
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    radii = n_std * np.sqrt(evals)
    region = (template * radii) @ evecs.T + mean
    ax.plot_surface(
        region[..., 0],
        region[..., 1],
        region[..., 2],
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        alpha=alpha,
        color=color,
    )
    ax.scatter(mean[0], mean[1], mean[2], color=color, s=42)


def _draw_direction_circle(ax, *, labels: tuple[str, str]) -> None:
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


def _make_parameter_axes(*, figsize: tuple[float, float], axes):
    if axes is None:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax_linear = fig.add_subplot(1, 2, 1, projection="3d")
        ax_direction = fig.add_subplot(1, 2, 2)
        return fig, ax_linear, ax_direction

    axes_arr = np.asarray(axes, dtype=object).ravel()
    if axes_arr.size != 2:
        raise ValueError("axes must contain exactly two axes.")
    ax_linear, ax_direction = axes_arr
    fig = ax_linear.figure
    if ax_direction.figure is not fig:
        raise ValueError("Both axes must belong to the same figure.")
    return fig, ax_linear, ax_direction


def _sample_rows(
        sample: np.ndarray,
        labels,
        *,
        max_points: int | None,
        random_state: int | np.random.RandomState | None,
        ) -> tuple[np.ndarray, np.ndarray | None]:
    n_obs = sample.shape[0]
    if max_points is None:
        return sample, labels

    if not isinstance(max_points, (int, np.integer)) or int(max_points) < 1:
        raise ValueError("max_points must be an integer >= 1.")
    max_points = int(max_points)
    if n_obs <= max_points:
        return sample, labels

    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)
    idx = np.sort(rng.choice(n_obs, size=max_points, replace=False))
    sample = sample[idx]
    if labels is not None:
        labels = labels[idx]
    return sample, labels


def _normalize_direction_sample(directions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms[:, 0] <= 0.0):
        raise ValueError("direction columns must not contain zero vectors.")
    return directions / norms


def plot_gaussian_vmf_sample(
        sample,
        *,
        labels=None,
        linear_feature_names: tuple[str, str, str] = ("wvht", "apd", "wspd"),
        direction_feature_names: tuple[str, str] = ("cos(mwd)", "sin(mwd)"),
        figsize: tuple[float, float] = (11.0, 5.0),
        point_size: float = 14.0,
        point_alpha: float = 0.58,
        max_points: int | None = None,
        random_state: int | np.random.RandomState | None = 57,
        normalize_directions: bool = True,
        legend: bool = True,
        show: bool = True,
        title: str | None = "Gaussian/vMF sample",
        axes=None,
        save_path=None,
        save_kwargs: Mapping[str, Any] | None = None,
        light_background: bool = True,
        ):
    """
    Plot sample observations as 3D Gaussian points and unit-circle vMF points.

    ``sample`` must have shape ``(N, 5)``: the first three columns are the
    Euclidean block and the last two columns are the circular direction block.
    """
    import matplotlib.pyplot as plt

    sample = np.asarray(sample, dtype=float)
    if sample.ndim != 2 or sample.shape[1] != 5:
        raise ValueError("sample must have shape (N, 5).")
    if not np.all(np.isfinite(sample)):
        raise ValueError("sample contains non-finite values.")

    n_obs = sample.shape[0]
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.shape[0] != n_obs:
            raise ValueError("labels must have shape (N,).")

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
    point_size = _validate_positive_float(point_size, name="point_size")
    point_alpha = _validate_unit_interval(point_alpha, name="point_alpha")

    sample, labels = _sample_rows(
        sample,
        labels,
        max_points=max_points,
        random_state=random_state,
    )
    linear = sample[:, :3]
    directions = sample[:, 3:]
    if normalize_directions:
        directions = _normalize_direction_sample(directions)

    fig, ax_linear, ax_direction = _make_parameter_axes(figsize=figsize, axes=axes)
    _draw_direction_circle(ax_direction, labels=direction_feature_names)

    cmap = plt.cm.tab10
    if labels is None:
        color = cmap(0)
        ax_linear.scatter(
            linear[:, 0],
            linear[:, 1],
            linear[:, 2],
            color=color,
            alpha=point_alpha,
            s=point_size,
            linewidths=0.0,
        )
        ax_direction.scatter(
            directions[:, 0],
            directions[:, 1],
            color=color,
            alpha=point_alpha,
            s=point_size,
            linewidths=0.0,
            zorder=3,
        )
    else:
        for idx, value in enumerate(np.unique(labels)):
            mask = labels == value
            color = cmap(idx % 10)
            label = str(value)
            ax_linear.scatter(
                linear[mask, 0],
                linear[mask, 1],
                linear[mask, 2],
                color=color,
                alpha=point_alpha,
                s=point_size,
                linewidths=0.0,
                label=label,
            )
            ax_direction.scatter(
                directions[mask, 0],
                directions[mask, 1],
                color=color,
                alpha=point_alpha,
                s=point_size,
                linewidths=0.0,
                label=label,
                zorder=3,
            )

        if legend:
            ax_direction.legend(title="label", frameon=False, loc="upper right")

    ax_linear.set_xlabel(linear_feature_names[0])
    ax_linear.set_ylabel(linear_feature_names[1])
    ax_linear.set_zlabel(linear_feature_names[2])
    ax_linear.set_title("3D Euclidean sample")
    _set_axes_equal_3d(ax_linear)

    ax_direction.set_title("Circular sample")
    if title is not None:
        fig.suptitle(str(title))

    _finalize_figure(
        fig,
        show=show,
        save_path=save_path,
        save_kwargs=save_kwargs,
        light_background=light_background,
    )

    return fig, np.asarray([ax_linear, ax_direction], dtype=object)


def _extract_cross_covariance_matrix(
        component,
        *,
        idx: int,
        shape: tuple[int, int],
        ) -> np.ndarray:
    if not hasattr(component, "cross_cov"):
        raise ValueError(f"components[{idx}] must expose a cross_cov matrix.")
    matrix = np.asarray(component.cross_cov, dtype=float)
    if matrix.shape != shape:
        raise ValueError(
            f"components[{idx}].cross_cov must have shape {shape}, got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"components[{idx}].cross_cov contains non-finite values.")
    return matrix


def _resolve_heatmap_limits(
        matrices: list[np.ndarray],
        *,
        center_zero: bool,
        vmin: float | None,
        vmax: float | None,
        ) -> tuple[float, float]:
    values = np.concatenate([matrix.ravel() for matrix in matrices])
    data_min = float(np.min(values))
    data_max = float(np.max(values))

    if vmin is None:
        vmin = data_min
    else:
        vmin = float(vmin)
    if vmax is None:
        vmax = data_max
    else:
        vmax = float(vmax)

    if center_zero:
        bound = max(abs(vmin), abs(vmax))
        if not np.isfinite(bound) or bound <= 0.0:
            bound = 1.0
        vmin = -bound
        vmax = bound

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError("vmin and vmax must be finite with vmin < vmax.")
    return float(vmin), float(vmax)


def plot_cylindrical_cross_covariance_heatmaps(
        model_or_result,
        *,
        linear_feature_names: tuple[str, str, str] = ("wvht", "apd", "wspd"),
        direction_feature_names: tuple[str, str] = ("cos(mwd)", "sin(mwd)"),
        figsize: tuple[float, float] | None = None,
        max_cols: int = 4,
        cmap: str = "coolwarm",
        center_zero: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        annotate: bool = True,
        annotation_format: str = ".3g",
        colorbar: bool = True,
        show: bool = True,
        title: str | None = "Cylindrical cross covariance",
        axes=None,
        save_path=None,
        save_kwargs: Mapping[str, Any] | None = None,
        light_background: bool = True,
        ):
    """
    Plot per-component cylindrical cross-covariance matrices as heatmaps.

    ``model_or_result`` may be a fitted cylindrical ``MixtureModel`` or the
    dictionary returned by ``grid_search_cylindrical_mixture_bic``. With the
    default labels, each component must expose a ``3 x 2`` ``cross_cov`` matrix.
    """
    import matplotlib.pyplot as plt

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
    if not isinstance(max_cols, (int, np.integer)) or int(max_cols) < 1:
        raise ValueError("max_cols must be an integer >= 1.")
    max_cols = int(max_cols)

    _, components = _cylindrical_model(model_or_result, name="model_or_result")
    shape = (len(linear_feature_names), len(direction_feature_names))
    matrices = [
        _extract_cross_covariance_matrix(component, idx=idx, shape=shape)
        for idx, component in enumerate(components)
    ]
    vmin, vmax = _resolve_heatmap_limits(
        matrices,
        center_zero=bool(center_zero),
        vmin=vmin,
        vmax=vmax,
    )

    n_components = len(matrices)
    n_cols = min(max_cols, n_components)
    n_rows = int(np.ceil(n_components / n_cols))
    if figsize is None:
        figsize = (3.1 * n_cols, 2.9 * n_rows)

    if axes is None:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
        )
    else:
        axes = np.asarray(axes, dtype=object)
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        if axes.ndim != 2:
            raise ValueError("axes must be a 1D or 2D array of matplotlib axes.")
        if axes.size < n_components:
            raise ValueError(f"axes must contain at least {n_components} axes.")
        fig = axes.ravel()[0].figure
        if any(ax.figure is not fig for ax in axes.ravel()[:n_components]):
            raise ValueError("All axes must belong to the same figure.")

    im = None
    axes_flat = axes.ravel()
    text_threshold = 0.5 * max(abs(vmin), abs(vmax))
    for idx, matrix in enumerate(matrices):
        ax = axes_flat[idx]
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"C{idx}")
        ax.set_xticks(np.arange(len(direction_feature_names)))
        ax.set_xticklabels(direction_feature_names)
        ax.set_yticks(np.arange(len(linear_feature_names)))
        ax.set_yticklabels(linear_feature_names)
        ax.set_xlabel("Directional")
        ax.set_ylabel("Linear")

        if annotate:
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value = float(matrix[row, col])
                    text_color = "white" if abs(value) >= text_threshold else "black"
                    ax.text(
                        col,
                        row,
                        format(value, annotation_format),
                        ha="center",
                        va="center",
                        color=text_color,
                    )

    for idx in range(n_components, axes.size):
        axes_flat[idx].axis("off")

    if colorbar and im is not None:
        fig.colorbar(
            im,
            ax=axes_flat[:n_components].tolist(),
            shrink=0.85,
            label="Cross covariance",
        )

    if title is not None:
        fig.suptitle(str(title))

    _finalize_figure(
        fig,
        show=show,
        save_path=save_path,
        save_kwargs=save_kwargs,
        light_background=light_background,
    )

    return fig, axes


def _plot_two_layer_gaussian_vmf_parameters(
        model_or_result,
        *,
        linear_feature_names: tuple[str, str, str] = ("wvht", "apd", "wspd"),
        direction_feature_names: tuple[str, str] = ("cos(mwd)", "sin(mwd)"),
        figsize: tuple[float, float] = (11.0, 5.0),
        n_std: float = 1.5,
        n_ellipsoid: int = 36,
        ellipsoid_alpha: float = 0.22,
        direction_arrow_scale: float = 0.9,
        normalize_kappa: bool = True,
        scale_by_layer2_weight: bool = False,
        legend: bool = True,
        show: bool = True,
        title: str | None = "Two-layer mixture parameters",
        axes=None,
        save_path=None,
        save_kwargs: Mapping[str, Any] | None = None,
        light_background: bool = True,
        ):
    """
    Plot a two-layer model with 3D Gaussian layer-1 and 2D vMF layer-2 components.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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
    n_std = _validate_positive_float(n_std, name="n_std")
    direction_arrow_scale = _validate_positive_float(
        direction_arrow_scale,
        name="direction_arrow_scale",
    )
    ellipsoid_alpha = _validate_unit_interval(ellipsoid_alpha, name="ellipsoid_alpha")
    template = _ellipsoid_template(n_ellipsoid)

    _, layer1_mixture, layer1_components, layer2_mixtures = _two_layer_model(
        model_or_result,
        name="model_or_result",
    )
    layer1_weights = _mixture_weights(
        layer1_mixture,
        n_components=len(layer1_components),
        name="model_or_result.layer1_mixture",
    )
    layer1_params = [
        _extract_gaussian_component_params(
            component,
            idx=idx,
            size=3,
            name="layer1_mixture.components",
        )
        for idx, component in enumerate(layer1_components)
    ]

    layer2_items: list[dict[str, Any]] = []
    for layer1_idx, layer2_mixture in enumerate(layer2_mixtures):
        layer2_components = _mixture_components(
            layer2_mixture,
            name=f"layer2_mixtures[{layer1_idx}]",
        )
        layer2_weights = _mixture_weights(
            layer2_mixture,
            n_components=len(layer2_components),
            name=f"layer2_mixtures[{layer1_idx}]",
        )
        for layer2_idx, (component, layer2_weight) in enumerate(
                zip(layer2_components, layer2_weights)
        ):
            direction, kappa = _extract_vmf_component_params(
                component,
                size=2,
                name=f"layer2_mixtures[{layer1_idx}].components[{layer2_idx}]",
            )
            layer2_items.append({
                "layer1_idx": layer1_idx,
                "layer2_idx": layer2_idx,
                "layer1_weight": float(layer1_weights[layer1_idx]),
                "layer2_weight": float(layer2_weight),
                "direction": direction,
                "kappa": kappa,
            })

    if not layer2_items:
        raise ValueError("No layer-2 components to plot.")

    fig, ax_linear, ax_direction = _make_parameter_axes(figsize=figsize, axes=axes)
    _draw_direction_circle(ax_direction, labels=direction_feature_names)

    kappas = np.asarray([item["kappa"] for item in layer2_items], dtype=float)
    kappa_ref = float(np.max(kappas)) if normalize_kappa else 1.0
    if not np.isfinite(kappa_ref) or kappa_ref <= 0.0:
        kappa_ref = 1.0

    cmap = plt.cm.tab10
    for layer1_idx, ((mean, cov), layer1_weight) in enumerate(
            zip(layer1_params, layer1_weights)
    ):
        color = cmap(layer1_idx % 10)
        _plot_gaussian_ellipsoid(
            ax_linear,
            template,
            mean,
            cov,
            n_std=n_std,
            color=color,
            alpha=ellipsoid_alpha,
        )
        ax_linear.text(
            mean[0],
            mean[1],
            mean[2],
            f"L1 {layer1_idx}",
            color=color,
        )

    legend_handles = []
    for item in layer2_items:
        color = cmap(item["layer1_idx"] % 10)
        direction = item["direction"]
        arrow_length = direction_arrow_scale * item["kappa"] / kappa_ref
        if scale_by_layer2_weight:
            arrow_length *= item["layer2_weight"]

        ax_direction.quiver(
            0.0,
            0.0,
            arrow_length * direction[0],
            arrow_length * direction[1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=0.006,
            linewidth=1.0 + 1.5 * item["layer2_weight"],
            alpha=0.45 + 0.5 * item["layer2_weight"],
            zorder=3,
        )
        ax_direction.scatter(
            [direction[0]],
            [direction[1]],
            color=color,
            s=30 + 40 * item["layer2_weight"],
            zorder=4,
        )
        label = f"L1 {item['layer1_idx']}, L2 {item['layer2_idx']}"
        ax_direction.text(
            1.08 * direction[0],
            1.08 * direction[1],
            f"{item['layer1_idx']}.{item['layer2_idx']}",
            color=color,
            ha="center",
            va="center",
            fontsize=9,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                linestyle="",
                label=(
                    f"{label}: w1={item['layer1_weight']:.3g}, "
                    f"w2={item['layer2_weight']:.3g}, "
                    f"kappa={item['kappa']:.3g}"
                ),
            )
        )

    ax_linear.set_xlabel(linear_feature_names[0])
    ax_linear.set_ylabel(linear_feature_names[1])
    ax_linear.set_zlabel(linear_feature_names[2])
    ax_linear.set_title("Layer-1 Gaussian components")
    _set_axes_equal_3d(ax_linear)

    ax_direction.set_title("Layer-2 vMF directions")
    if legend:
        ax_direction.legend(handles=legend_handles, frameon=False, loc="upper right")

    if title is not None:
        fig.suptitle(str(title))

    _finalize_figure(
        fig,
        show=show,
        save_path=save_path,
        save_kwargs=save_kwargs,
        light_background=light_background,
    )

    return fig, np.asarray([ax_linear, ax_direction], dtype=object)


def _plot_two_layer_vmf_gaussian_parameters(
        model_or_result,
        *,
        linear_feature_names: tuple[str, str, str] = ("wvht", "apd", "wspd"),
        direction_feature_names: tuple[str, str] = ("cos(mwd)", "sin(mwd)"),
        figsize: tuple[float, float] = (11.0, 5.0),
        n_std: float = 1.5,
        n_ellipsoid: int = 36,
        ellipsoid_alpha: float = 0.22,
        direction_arrow_scale: float = 0.9,
        normalize_kappa: bool = True,
        scale_by_layer2_weight: bool = False,
        legend: bool = True,
        show: bool = True,
        title: str | None = "Two-layer mixture parameters",
        axes=None,
        save_path=None,
        save_kwargs: Mapping[str, Any] | None = None,
        light_background: bool = True,
        ):
    """Plot a two-layer model with 2D vMF layer-1 and 3D Gaussian layer-2 components."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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
    n_std = _validate_positive_float(n_std, name="n_std")
    direction_arrow_scale = _validate_positive_float(
        direction_arrow_scale,
        name="direction_arrow_scale",
    )
    ellipsoid_alpha = _validate_unit_interval(ellipsoid_alpha, name="ellipsoid_alpha")
    template = _ellipsoid_template(n_ellipsoid)

    _, layer1_mixture, layer1_components, layer2_mixtures = _two_layer_model(
        model_or_result,
        name="model_or_result",
    )
    layer1_weights = _mixture_weights(
        layer1_mixture,
        n_components=len(layer1_components),
        name="model_or_result.layer1_mixture",
    )
    layer1_params = [
        _extract_vmf_component_params(
            component,
            size=2,
            name=f"layer1_mixture.components[{idx}]",
        )
        for idx, component in enumerate(layer1_components)
    ]

    layer2_items: list[dict[str, Any]] = []
    for layer1_idx, layer2_mixture in enumerate(layer2_mixtures):
        layer2_components = _mixture_components(
            layer2_mixture,
            name=f"layer2_mixtures[{layer1_idx}]",
        )
        layer2_weights = _mixture_weights(
            layer2_mixture,
            n_components=len(layer2_components),
            name=f"layer2_mixtures[{layer1_idx}]",
        )
        for layer2_idx, (component, layer2_weight) in enumerate(
                zip(layer2_components, layer2_weights)
        ):
            mean, cov = _extract_gaussian_component_params(
                component,
                idx=layer2_idx,
                size=3,
                name=f"layer2_mixtures[{layer1_idx}].components",
            )
            layer2_items.append({
                "layer1_idx": layer1_idx,
                "layer2_idx": layer2_idx,
                "layer1_weight": float(layer1_weights[layer1_idx]),
                "layer2_weight": float(layer2_weight),
                "mean": mean,
                "cov": cov,
            })

    if not layer2_items:
        raise ValueError("No layer-2 components to plot.")

    fig, ax_linear, ax_direction = _make_parameter_axes(figsize=figsize, axes=axes)
    _draw_direction_circle(ax_direction, labels=direction_feature_names)

    kappas = np.asarray([kappa for _, kappa in layer1_params], dtype=float)
    kappa_ref = float(np.max(kappas)) if normalize_kappa else 1.0
    if not np.isfinite(kappa_ref) or kappa_ref <= 0.0:
        kappa_ref = 1.0

    cmap = plt.cm.tab10
    direction_legend_handles = []
    for layer1_idx, ((direction, kappa), layer1_weight) in enumerate(
            zip(layer1_params, layer1_weights)
    ):
        color = cmap(layer1_idx % 10)
        arrow_length = direction_arrow_scale * kappa / kappa_ref
        ax_direction.quiver(
            0.0,
            0.0,
            arrow_length * direction[0],
            arrow_length * direction[1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=0.008,
            alpha=0.85,
            zorder=3,
        )
        ax_direction.scatter(
            [direction[0]],
            [direction[1]],
            color=color,
            s=35 + 80 * layer1_weight,
            zorder=4,
        )
        ax_direction.text(
            1.08 * direction[0],
            1.08 * direction[1],
            f"L1 {layer1_idx}",
            color=color,
            ha="center",
            va="center",
            fontsize=9,
        )
        direction_legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                linestyle="",
                label=(
                    f"L1 {layer1_idx}: "
                    f"w1={layer1_weight:.3g}, "
                    f"kappa={kappa:.3g}"
                ),
            )
        )

    for item in layer2_items:
        color = cmap(item["layer1_idx"] % 10)
        alpha = ellipsoid_alpha
        if scale_by_layer2_weight:
            alpha *= item["layer2_weight"]
        alpha = float(np.clip(alpha, 0.04, 1.0))

        _plot_gaussian_ellipsoid(
            ax_linear,
            template,
            item["mean"],
            item["cov"],
            n_std=n_std,
            color=color,
            alpha=alpha,
        )
        ax_linear.text(
            item["mean"][0],
            item["mean"][1],
            item["mean"][2],
            f"{item['layer1_idx']}.{item['layer2_idx']}",
            color=color,
        )

    ax_linear.set_xlabel(linear_feature_names[0])
    ax_linear.set_ylabel(linear_feature_names[1])
    ax_linear.set_zlabel(linear_feature_names[2])
    ax_linear.set_title("Layer-2 Gaussian components")
    _set_axes_equal_3d(ax_linear)

    ax_direction.set_title("Layer-1 vMF directions")
    if legend:
        ax_direction.legend(
            handles=direction_legend_handles,
            frameon=False,
            loc="upper right",
        )

    if title is not None:
        fig.suptitle(str(title))

    _finalize_figure(
        fig,
        show=show,
        save_path=save_path,
        save_kwargs=save_kwargs,
        light_background=light_background,
    )

    return fig, np.asarray([ax_linear, ax_direction], dtype=object)


def plot_two_layer_vmf_gaussian_parameters(
        model_or_result,
        *,
        linear_feature_names: tuple[str, str, str] = ("wvht", "apd", "wspd"),
        direction_feature_names: tuple[str, str] = ("cos(mwd)", "sin(mwd)"),
        figsize: tuple[float, float] = (11.0, 5.0),
        n_std: float = 1.5,
        n_ellipsoid: int = 36,
        ellipsoid_alpha: float = 0.22,
        direction_arrow_scale: float = 0.9,
        normalize_kappa: bool = True,
        scale_by_layer2_weight: bool = False,
        legend: bool = True,
        show: bool = True,
        title: str | None = "Two-layer mixture parameters",
        axes=None,
        save_path=None,
        save_kwargs: Mapping[str, Any] | None = None,
        light_background: bool = True,
        ):
    """
    Plot a vMF->Gaussian two-layer model with a layer-1 vMF direction legend.

    ``model_or_result`` may be a fitted ``TwoLayerMoM`` or a grid-search result
    dictionary with ``best_model``. The supported feature layout is the 2D
    directional block in layer 1 and 3D linear block in layer 2.
    """
    _, _, layer1_components, layer2_mixtures = _two_layer_model(
        model_or_result,
        name="model_or_result",
    )
    family = _infer_two_layer_family(layer1_components, layer2_mixtures)
    if family != ("vmf", "gaussian"):
        raise ValueError(
            "plot_two_layer_vmf_gaussian_parameters expects a vMF->Gaussian "
            "two-layer model."
        )

    return _plot_two_layer_vmf_gaussian_parameters(
        model_or_result,
        linear_feature_names=linear_feature_names,
        direction_feature_names=direction_feature_names,
        figsize=figsize,
        n_std=n_std,
        n_ellipsoid=n_ellipsoid,
        ellipsoid_alpha=ellipsoid_alpha,
        direction_arrow_scale=direction_arrow_scale,
        normalize_kappa=normalize_kappa,
        scale_by_layer2_weight=scale_by_layer2_weight,
        legend=legend,
        show=show,
        title=title,
        axes=axes,
        save_path=save_path,
        save_kwargs=save_kwargs,
        light_background=light_background,
    )


def plot_two_layer_mixture_parameters(
        model_or_result,
        *,
        linear_feature_names: tuple[str, str, str] = ("wvht", "apd", "wspd"),
        direction_feature_names: tuple[str, str] = ("cos(mwd)", "sin(mwd)"),
        figsize: tuple[float, float] = (11.0, 5.0),
        n_std: float = 1.5,
        n_ellipsoid: int = 36,
        ellipsoid_alpha: float = 0.22,
        direction_arrow_scale: float = 0.9,
        normalize_kappa: bool = True,
        scale_by_layer2_weight: bool = False,
        legend: bool = True,
        show: bool = True,
        title: str | None = "Two-layer mixture parameters",
        axes=None,
        save_path=None,
        save_kwargs: Mapping[str, Any] | None = None,
        light_background: bool = True,
        ):
    """
    Plot a two-layer model with either Gaussian->vMF or vMF->Gaussian components.

    ``model_or_result`` may be a fitted ``TwoLayerMoM`` or a grid-search result
    dictionary with ``best_model``. The supported feature layout is still the
    3D linear block and 2D directional block used by this experiment.
    """
    _, _, layer1_components, layer2_mixtures = _two_layer_model(
        model_or_result,
        name="model_or_result",
    )
    family = _infer_two_layer_family(layer1_components, layer2_mixtures)

    plot_kwargs = {
        "linear_feature_names": linear_feature_names,
        "direction_feature_names": direction_feature_names,
        "figsize": figsize,
        "n_std": n_std,
        "n_ellipsoid": n_ellipsoid,
        "ellipsoid_alpha": ellipsoid_alpha,
        "direction_arrow_scale": direction_arrow_scale,
        "normalize_kappa": normalize_kappa,
        "scale_by_layer2_weight": scale_by_layer2_weight,
        "legend": legend,
        "show": show,
        "title": title,
        "axes": axes,
        "save_path": save_path,
        "save_kwargs": save_kwargs,
        "light_background": light_background,
    }
    if family == ("gaussian", "vmf"):
        return _plot_two_layer_gaussian_vmf_parameters(model_or_result, **plot_kwargs)
    return _plot_two_layer_vmf_gaussian_parameters(model_or_result, **plot_kwargs)


def plot_cylindrical_mixture_parameters(
        model_or_result,
        *,
        linear_feature_names: tuple[str, str, str] = ("wvht", "apd", "wspd"),
        direction_feature_names: tuple[str, str] = ("cos(mwd)", "sin(mwd)"),
        covariance: str = "unconditional",
        figsize: tuple[float, float] = (11.0, 5.0),
        n_std: float = 1.5,
        n_ellipsoid: int = 36,
        ellipsoid_alpha: float = 0.22,
        direction_arrow_scale: float = 0.9,
        normalize_kappa: bool = True,
        legend: bool = True,
        show: bool = True,
        title: str | None = "Cylindrical mixture parameters",
        axes=None,
        save_path=None,
        save_kwargs: Mapping[str, Any] | None = None,
        light_background: bool = True,
        ):
    """
    Plot a 3D-linear, 2D-directional cylindrical mixture.

    ``model_or_result`` may be a fitted cylindrical ``MixtureModel`` or the
    dictionary returned by ``grid_search_cylindrical_mixture_bic``. Components
    must have ``d_gauss=3`` and ``d_vmf=2``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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
    covariance = _validate_cylindrical_covariance(covariance)
    n_std = _validate_positive_float(n_std, name="n_std")
    direction_arrow_scale = _validate_positive_float(
        direction_arrow_scale,
        name="direction_arrow_scale",
    )
    ellipsoid_alpha = _validate_unit_interval(ellipsoid_alpha, name="ellipsoid_alpha")
    template = _ellipsoid_template(n_ellipsoid)

    model, components = _cylindrical_model(model_or_result, name="model_or_result")
    weights = _mixture_weights(
        model,
        n_components=len(components),
        name="model_or_result",
    )
    component_params = [
        _extract_cylindrical_component_params(
            component,
            idx=idx,
            covariance=covariance,
        )
        for idx, component in enumerate(components)
    ]

    if axes is None:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax_linear = fig.add_subplot(1, 2, 1, projection="3d")
        ax_direction = fig.add_subplot(1, 2, 2)
    else:
        axes_arr = np.asarray(axes, dtype=object).ravel()
        if axes_arr.size != 2:
            raise ValueError("axes must contain exactly two axes.")
        ax_linear, ax_direction = axes_arr
        fig = ax_linear.figure
        if ax_direction.figure is not fig:
            raise ValueError("Both axes must belong to the same figure.")

    kappas = np.asarray([params["kappa"] for params in component_params], dtype=float)
    kappa_ref = float(np.max(kappas)) if normalize_kappa else 1.0
    if not np.isfinite(kappa_ref) or kappa_ref <= 0.0:
        kappa_ref = 1.0

    cmap = plt.cm.tab10
    legend_handles = []
    for idx, (weight, params) in enumerate(zip(weights, component_params)):
        color = cmap(idx % 10)
        _plot_gaussian_ellipsoid(
            ax_linear,
            template,
            params["mean"],
            params["cov"],
            n_std=n_std,
            color=color,
            alpha=ellipsoid_alpha,
        )
        ax_linear.text(
            params["mean"][0],
            params["mean"][1],
            params["mean"][2],
            f"C{idx}",
            color=color,
        )

        direction = params["direction"]
        arrow_length = direction_arrow_scale * params["kappa"] / kappa_ref
        ax_direction.quiver(
            0.0,
            0.0,
            arrow_length * direction[0],
            arrow_length * direction[1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=0.008,
            alpha=0.85,
            zorder=3,
        )
        ax_direction.scatter(
            [direction[0]],
            [direction[1]],
            color=color,
            s=32,
            zorder=4,
        )
        ax_direction.text(
            1.08 * direction[0],
            1.08 * direction[1],
            f"C{idx}",
            color=color,
            ha="center",
            va="center",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                linestyle="",
                label=f"C{idx}: w={weight:.3g}, kappa={params['kappa']:.3g}",
            )
        )

    ax_linear.set_xlabel(linear_feature_names[0])
    ax_linear.set_ylabel(linear_feature_names[1])
    ax_linear.set_zlabel(linear_feature_names[2])
    ax_linear.set_title(f"3D Gaussian ({covariance} covariance)")
    _set_axes_equal_3d(ax_linear)

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax_direction.plot(np.cos(theta), np.sin(theta), color="0.45", linewidth=1.0)
    ax_direction.axhline(0.0, color="0.85", linewidth=0.8)
    ax_direction.axvline(0.0, color="0.85", linewidth=0.8)
    ax_direction.set_xlabel(direction_feature_names[0])
    ax_direction.set_ylabel(direction_feature_names[1])
    ax_direction.set_title("2D vMF direction")
    ax_direction.set_xlim(-1.18, 1.18)
    ax_direction.set_ylim(-1.18, 1.18)
    ax_direction.set_aspect("equal", adjustable="box")
    ax_direction.grid(True, alpha=0.2)
    if legend:
        ax_direction.legend(handles=legend_handles, frameon=False, loc="upper right")

    if title is not None:
        fig.suptitle(str(title))

    _finalize_figure(
        fig,
        show=show,
        save_path=save_path,
        save_kwargs=save_kwargs,
        light_background=light_background,
    )

    return fig, np.asarray([ax_linear, ax_direction], dtype=object)


def plot_cylindrical_grid_search_bic(
        search_result,
        *,
        metric: str = "bic_min",
        ax=None,
        figsize: tuple[float, float] = (6.0, 4.0),
        marker: str = "o",
        title: str | None = "Cylindrical Mixture BIC Grid Search",
        show: bool = True,
        annotate_best: bool = True,
        save_path=None,
        save_kwargs: Mapping[str, Any] | None = None,
        light_background: bool = True,
        ):
    """
    Plot cylindrical-mixture grid-search BIC as a 2D curve over ``K``.

    ``search_result`` may be the dictionary returned by
    ``grid_search_cylindrical_mixture_bic`` or its ``config_results`` list.
    """
    import matplotlib.pyplot as plt

    rows = _extract_config_results(search_result, name="search_result")
    rows = _finite_successful_config_rows(rows, required_keys=("k",), metric=metric)
    rows = sorted(rows, key=lambda row: int(row["k"]))

    k_values = np.asarray([int(row["k"]) for row in rows], dtype=int)
    metric_values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    best_idx = int(np.argmin(metric_values))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(k_values, metric_values, marker=marker, linewidth=1.8)
    ax.scatter(
        [k_values[best_idx]],
        [metric_values[best_idx]],
        s=70,
        color="tab:red",
        zorder=3,
    )
    ax.set_xlabel("K")
    ax.set_ylabel(metric.replace("_", " ").upper())
    if title is not None:
        ax.set_title(str(title))
    ax.set_xticks(k_values)
    ax.grid(alpha=0.25)

    if annotate_best:
        ax.annotate(
            f"K={k_values[best_idx]}",
            xy=(k_values[best_idx], metric_values[best_idx]),
            xytext=(6, 8),
            textcoords="offset points",
        )

    _finalize_figure(
        fig,
        show=show,
        save_path=save_path,
        save_kwargs=save_kwargs,
        light_background=light_background,
    )

    return fig, ax


def plot_two_layer_grid_search_bic_surface(
        search_result,
        *,
        metric: str = "bic_min",
        ax=None,
        figsize: tuple[float, float] = (7.0, 5.0),
        cmap: str = "viridis",
        colorbar: bool = True,
        title: str | None = "Two-Layer Mixture BIC Grid Search",
        show: bool = True,
        annotate_best: bool = True,
        elev: float | None = None,
        azim: float | None = None,
        save_path=None,
        save_kwargs: Mapping[str, Any] | None = None,
        light_background: bool = True,
        ):
    """
    Plot two-layer Gaussian/vMF grid-search BIC as a 3D surface over ``(K, L)``.

    ``search_result`` may be the dictionary returned by
    ``grid_search_two_layer_mixture_bic`` or its ``config_results`` list.
    Use ``elev`` and ``azim`` to set the 3D viewing angle in degrees.
    """
    import matplotlib.pyplot as plt

    rows = _extract_config_results(search_result, name="search_result")
    rows = _finite_successful_config_rows(rows, required_keys=("k", "l"), metric=metric)

    k_values = np.asarray(sorted({int(row["k"]) for row in rows}), dtype=int)
    l_values = np.asarray(sorted({int(row["l"]) for row in rows}), dtype=int)
    z_values = np.full((l_values.size, k_values.size), np.nan, dtype=float)
    k_index = {value: idx for idx, value in enumerate(k_values)}
    l_index = {value: idx for idx, value in enumerate(l_values)}

    for row in rows:
        z_values[l_index[int(row["l"])], k_index[int(row["k"])]] = float(row[metric])

    kk, ll = np.meshgrid(k_values, l_values)
    finite_mask = np.isfinite(z_values)
    best_flat_idx = int(np.nanargmin(z_values))
    best_l_idx, best_k_idx = np.unravel_index(best_flat_idx, z_values.shape)
    best_k = int(k_values[best_k_idx])
    best_l = int(l_values[best_l_idx])
    best_metric = float(z_values[best_l_idx, best_k_idx])

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    surface = ax.plot_surface(
        kk,
        ll,
        np.ma.masked_invalid(z_values),
        cmap=cmap,
        edgecolor="0.35",
        linewidth=0.4,
        alpha=0.9,
    )
    ax.scatter(
        kk[finite_mask],
        ll[finite_mask],
        z_values[finite_mask],
        color="black",
        s=20,
        depthshade=False,
    )
    ax.scatter(
        [best_k],
        [best_l],
        [best_metric],
        color="tab:red",
        s=70,
        depthshade=False,
    )

    ax.set_xlabel("K")
    ax.set_ylabel("L")
    ax.set_zlabel(metric.replace("_", " ").upper())
    if title is not None:
        ax.set_title(str(title))
    ax.set_xticks(k_values)
    ax.set_yticks(l_values)
    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)

    if annotate_best:
        ax.text(best_k, best_l, best_metric, f" K={best_k}, L={best_l}", color="tab:red")

    if colorbar:
        fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.12, label=metric.replace("_", " ").upper())

    _finalize_figure(
        fig,
        show=show,
        save_path=save_path,
        save_kwargs=save_kwargs,
        light_background=light_background,
    )

    return fig, ax


__all__ = [
    "plot_gaussian_vmf_sample",
    "plot_cylindrical_cross_covariance_heatmaps",
    "plot_cylindrical_grid_search_bic",
    "plot_cylindrical_mixture_parameters",
    "plot_two_layer_mixture_parameters",
    "plot_two_layer_vmf_gaussian_parameters",
    "plot_two_layer_grid_search_bic_surface",
]
