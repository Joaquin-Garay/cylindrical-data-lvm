"""Training and model-selection helpers for ocean experiments."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

import cyl_lvm as clvm


def _validate_component_grid(name: str, values: Sequence[int]) -> list[int]:
    values = list(values)
    if not values:
        raise ValueError(f"{name} must contain at least one value.")

    out: list[int] = []
    for value in values:
        if not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise ValueError(f"All values in {name} must be integers >= 1.")
        out.append(int(value))
    return out


def _validate_n_restarts(n_restarts: int) -> int:
    if not isinstance(n_restarts, (int, np.integer)) or int(n_restarts) < 1:
        raise ValueError("n_restarts must be an integer >= 1.")
    return int(n_restarts)


def _validate_positive_int(value: int, *, name: str, minimum: int = 1) -> int:
    if not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _validate_2d_array(x, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _resolve_rng(random_state: Optional[int | np.random.RandomState]) -> np.random.RandomState:
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(random_state)


def _next_seed(rng: np.random.RandomState) -> int:
    return int(rng.randint(np.iinfo(np.int32).max))


def _format_bic(value: float) -> str:
    value = float(value)
    if np.isfinite(value):
        return f"{value:.6g}"
    return str(value)


def _format_metric(value: float) -> str:
    value = float(value)
    if np.isfinite(value):
        return f"{value:.6g}"
    return str(value)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


def _array_stats(value) -> dict[str, Any]:
    arr = np.asarray(value, dtype=float)
    return {
        "shape": tuple(int(dim) for dim in arr.shape),
        "min": float(np.min(arr)) if arr.size else np.nan,
        "max": float(np.max(arr)) if arr.size else np.nan,
        "mean": float(np.mean(arr)) if arr.size else np.nan,
        "fro_norm": float(np.linalg.norm(arr.ravel())) if arr.size else 0.0,
    }


def _safe_getattr(obj, name: str):
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _component_parameter_summary(component) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(component).__name__}
    for attr in (
            "mean",
            "covariance",
            "mu",
            "kappa",
            "mu_gauss",
            "cross_cov",
            "cross_corr",
            "cond_cov",
            "unconditional_gauss_cov",
            ):
        value = _safe_getattr(component, attr)
        if value is None:
            continue
        summary[attr] = _jsonable(value)
        if attr in {"cross_cov", "cross_corr", "cond_cov", "unconditional_gauss_cov"}:
            summary[f"{attr}_stats"] = _array_stats(value)

    gaussian = _safe_getattr(component, "gaussian")
    if gaussian is not None and gaussian is not component:
        summary["gaussian"] = _component_parameter_summary(gaussian)

    vmf = _safe_getattr(component, "vmf")
    if vmf is not None and vmf is not component:
        summary["vmf"] = _component_parameter_summary(vmf)

    return _jsonable(summary)


def _mixture_parameter_summary(model) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(model).__name__}
    n_components = _safe_getattr(model, "n_components")
    if n_components is not None:
        summary["n_components"] = int(n_components)

    weights = _safe_getattr(model, "weights")
    if weights is not None:
        summary["weights"] = _jsonable(np.asarray(weights, dtype=float))

    components = _safe_getattr(model, "components")
    if components is not None:
        summary["components"] = [
            _component_parameter_summary(component)
            for component in components
        ]
        try:
            summary["cross_covariance"] = _jsonable(
                _cylindrical_cross_covariance_summary(model)
            )
        except Exception:
            pass

    return _jsonable(summary)


def _model_parameter_summary(model) -> dict[str, Any]:
    if (
            _safe_getattr(model, "layer1_mixture") is not None
            and _safe_getattr(model, "layer2_mixtures") is not None
            ):
        return _jsonable({
            "type": type(model).__name__,
            "n_layer1_components": int(model.n_layer1_components),
            "layer1_mixture": _mixture_parameter_summary(model.layer1_mixture),
            "layer2_mixtures": [
                _mixture_parameter_summary(mixture)
                for mixture in model.layer2_mixtures
            ],
        })
    return _mixture_parameter_summary(model)


def _validate_start_seeds(
        start_seeds: Sequence[int],
        *,
        name: str = "start_seeds",
        ) -> list[int]:
    seeds = list(start_seeds)
    if not seeds:
        raise ValueError(f"{name} must contain at least one seed.")

    max_seed = int(np.iinfo(np.uint32).max)
    out: list[int] = []
    for seed in seeds:
        if (
                not isinstance(seed, (int, np.integer))
                or int(seed) < 0
                or int(seed) > max_seed
                ):
            raise ValueError(f"All values in {name} must be integer seeds in [0, {max_seed}].")
        out.append(int(seed))
    return out


def _resolve_start_seeds(
        *,
        n_starts: int,
        random_state: Optional[int | np.random.RandomState],
        start_seeds: Optional[Sequence[int]],
        ) -> list[int]:
    if start_seeds is not None:
        return _validate_start_seeds(start_seeds)

    n_starts = _validate_n_restarts(n_starts)
    master_rng = _resolve_rng(random_state)
    return [_next_seed(master_rng) for _ in range(n_starts)]


def _model_converged(model, *, max_iter: int) -> Optional[bool]:
    n_iter = _safe_getattr(model, "n_iter")
    if n_iter is None:
        return None
    return bool(int(n_iter) < int(max_iter) - 1)


def _validate_log_likelihood_vector(
        ll,
        *,
        n_obs: int,
        name: str,
        ) -> np.ndarray:
    values = np.asarray(ll, dtype=float)
    if values.ndim != 1 or values.shape[0] != n_obs:
        raise ValueError(
            f"{name} must be a 1D vector with length {n_obs}; got shape {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")
    return values


def _compute_fitted_model_metrics(
        model,
        *,
        train_args: tuple[np.ndarray, ...],
        test_args: tuple[np.ndarray, ...],
        max_iter: int,
        fit_time: float,
        ) -> dict[str, Any]:
    n_train = int(train_args[0].shape[0])
    n_test = int(test_args[0].shape[0])
    train_ll = _validate_log_likelihood_vector(
        model.log_pdf(*train_args),
        n_obs=n_train,
        name="training log-likelihood",
    )
    heldout_ll = _validate_log_likelihood_vector(
        model.log_pdf(*test_args),
        n_obs=n_test,
        name="held-out log-likelihood",
    )

    train_bic = float(model.bic_score(*train_args))
    train_aic = float(model.aic_score(*train_args))
    heldout_bic = float(model.bic_score(*test_args))
    heldout_aic = float(model.aic_score(*test_args))
    if not np.all(np.isfinite([train_bic, train_aic, heldout_bic, heldout_aic])):
        raise ValueError("AIC/BIC metric contains a non-finite value.")

    n_iter = _safe_getattr(model, "n_iter")
    n_free_params = None
    if hasattr(model, "n_free_params"):
        n_free_params = int(model.n_free_params())

    return {
        "n_train": n_train,
        "n_test": n_test,
        "train_log_likelihood": float(np.sum(train_ll)),
        "heldout_log_likelihood": float(np.sum(heldout_ll)),
        "train_avg_log_likelihood": float(np.mean(train_ll)),
        "heldout_avg_log_likelihood": float(np.mean(heldout_ll)),
        "train_bic": train_bic,
        "train_aic": train_aic,
        "heldout_bic": heldout_bic,
        "heldout_aic": heldout_aic,
        "train_gmpd": float(np.exp(np.mean(train_ll))),
        "heldout_gmpd": float(np.exp(np.mean(heldout_ll))),
        "n_free_params": n_free_params,
        "n_iter": None if n_iter is None else int(n_iter),
        "converged": _model_converged(model, max_iter=max_iter),
        "fit_time": float(fit_time),
        "parameter_summary": _model_parameter_summary(model),
    }


def _print_multistart_progress(
        *,
        completed_runs: int,
        total_runs: int,
        rec: dict[str, Any],
        descriptor: str,
        ) -> None:
    progress_pct = 100.0 * completed_runs / total_runs
    status = "success" if rec["success"] else "failed"
    message = (
        f"Finished EM start {completed_runs}/{total_runs} "
        f"({progress_pct:.1f}%) | {descriptor} | start={rec['em_start']} "
        f"| seed={rec['em_seed']} | train_BIC={_format_bic(rec['train_bic'])} "
        f"| heldout_LL={_format_metric(rec['heldout_log_likelihood'])} | {status}"
    )
    if not rec["success"] and "error" in rec:
        message = f"{message} | error={rec['error']}"
    print(message)


def _print_grid_search_progress(
        *,
        completed_runs: int,
        total_runs: int,
        rec: dict[str, Any],
        keys: tuple[str, ...],
        ) -> None:
    progress_pct = 100.0 * completed_runs / total_runs
    config = " ".join(f"{key}={rec[key]}" for key in keys)
    status = "success" if rec["success"] else "failed"
    message = (
        f"Finished run {completed_runs}/{total_runs} "
        f"({progress_pct:.1f}%) | {config} | restart={rec['restart']} "
        f"| BIC={_format_bic(rec['bic'])} | {status}"
    )
    if not rec["success"] and "error" in rec:
        message = f"{message} | error={rec['error']}"
    print(message)


def _print_dependence_progress(
        *,
        completed_runs: int,
        total_runs: int,
        rec: dict[str, Any],
        ) -> None:
    progress_pct = 100.0 * completed_runs / total_runs
    status = "success" if rec["success"] else "failed"
    message = (
        f"Finished paired run {completed_runs}/{total_runs} "
        f"({progress_pct:.1f}%) | k={rec['k']} | restart={rec['restart']} "
        f"| delta_hll={_format_metric(rec['delta_hll'])} "
        f"| delta_bic={_format_metric(rec['delta_bic'])} | {status}"
    )
    if not rec["success"] and "error" in rec:
        message = f"{message} | error={rec['error']}"
    print(message)


def _summarize_config_results(
        results: list[dict[str, Any]],
        *,
        keys: tuple[str, ...],
        candidates: Sequence[tuple[int, ...]],
        ) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        matching = [
            rec
            for rec in results
            if all(int(rec[key]) == value for key, value in zip(keys, candidate))
        ]
        successful = [rec for rec in matching if rec["success"]]
        base = {key: value for key, value in zip(keys, candidate)}
        if successful:
            bics = np.asarray([float(rec["bic"]) for rec in successful], dtype=float)
            best_run = min(successful, key=lambda rec: float(rec["bic"]))
            summaries.append({
                **base,
                "success": True,
                "successful_restarts": len(successful),
                "failed_restarts": len(matching) - len(successful),
                "bic_min": float(np.min(bics)),
                "bic_mean": float(np.mean(bics)),
                "bic_median": float(np.median(bics)),
                "bic_std": float(np.std(bics)),
                "best_restart": int(best_run["restart"]),
                "best_seed": int(best_run["seed"]),
                "best_n_iter": best_run["n_iter"],
            })
        else:
            summaries.append({
                **base,
                "success": False,
                "successful_restarts": 0,
                "failed_restarts": len(matching),
                "bic_min": np.inf,
                "bic_mean": np.inf,
                "bic_median": np.inf,
                "bic_std": np.inf,
                "best_restart": None,
                "best_seed": None,
                "best_n_iter": None,
            })
    return sorted(summaries, key=lambda rec: (not rec["success"], float(rec["bic_min"])))


def _validate_cylindrical_data(
        x_train,
        x_test,
        *,
        d_gauss: int,
        d_vmf: Optional[int],
        ) -> tuple[np.ndarray, np.ndarray, int, int]:
    x_train = _validate_2d_array(x_train, name="x_train")
    x_test = _validate_2d_array(x_test, name="x_test")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            "x_train and x_test must have the same number of features; "
            f"got {x_train.shape[1]} and {x_test.shape[1]}."
        )

    d_gauss = _validate_positive_int(d_gauss, name="d_gauss", minimum=1)
    if d_vmf is None:
        d_vmf = x_train.shape[1] - d_gauss
    d_vmf = _validate_positive_int(d_vmf, name="d_vmf", minimum=2)
    if x_train.shape[1] != d_gauss + d_vmf:
        raise ValueError(
            "feature dimension mismatch: expected d_gauss + d_vmf features "
            f"({d_gauss + d_vmf}), got {x_train.shape[1]}."
        )
    return x_train, x_test, d_gauss, d_vmf


def _validate_cylindrical_feature_array(
        x,
        *,
        d_gauss: int,
        d_vmf: Optional[int],
        name: str = "x",
        ) -> tuple[np.ndarray, int, int]:
    x = _validate_2d_array(x, name=name)
    d_gauss = _validate_positive_int(d_gauss, name="d_gauss", minimum=1)
    if d_vmf is None:
        d_vmf = x.shape[1] - d_gauss
    d_vmf = _validate_positive_int(d_vmf, name="d_vmf", minimum=2)
    if x.shape[1] != d_gauss + d_vmf:
        raise ValueError(
            f"{name} dimension mismatch: expected d_gauss + d_vmf features "
            f"({d_gauss + d_vmf}), got {x.shape[1]}."
        )
    return x, d_gauss, d_vmf


def _mixture_weights(model, *, n_components: int) -> np.ndarray:
    weights = np.asarray(model.weights, dtype=float)
    if weights.shape != (n_components,):
        raise ValueError(f"model weights must have shape ({n_components},), got {weights.shape}.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("model weights contain non-finite values.")
    if np.any(weights < 0.0):
        raise ValueError("model weights must be non-negative.")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("model weights must sum to a positive finite value.")
    return weights / total


def _summarize_dependence_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    k_values = sorted({int(rec["k"]) for rec in results})
    for k in k_values:
        matching = [rec for rec in results if int(rec["k"]) == k]
        successful = [rec for rec in matching if rec["success"]]
        if not successful:
            summaries.append({
                "k": k,
                "success": False,
                "successful_restarts": 0,
                "failed_restarts": len(matching),
            })
            continue

        delta_hll = np.asarray([float(rec["delta_hll"]) for rec in successful], dtype=float)
        delta_avg_hll = np.asarray(
            [float(rec["delta_avg_hll"]) for rec in successful],
            dtype=float,
        )
        delta_bic = np.asarray([float(rec["delta_bic"]) for rec in successful], dtype=float)
        delta_aic = np.asarray([float(rec["delta_aic"]) for rec in successful], dtype=float)
        delta_heldout_bic = np.asarray(
            [float(rec["delta_heldout_bic"]) for rec in successful],
            dtype=float,
        )
        delta_heldout_aic = np.asarray(
            [float(rec["delta_heldout_aic"]) for rec in successful],
            dtype=float,
        )
        cross_weighted = np.asarray(
            [float(rec["cross_cov_norm_weighted"]) for rec in successful],
            dtype=float,
        )
        cross_max = np.asarray(
            [float(rec["cross_cov_norm_max"]) for rec in successful],
            dtype=float,
        )
        best_run = min(successful, key=lambda rec: float(rec["bic_cyl"]))
        summary = {
            "k": k,
            "success": True,
            "successful_restarts": len(successful),
            "failed_restarts": len(matching) - len(successful),
            "delta_hll_mean": float(np.mean(delta_hll)),
            "delta_hll_median": float(np.median(delta_hll)),
            "delta_hll_std": float(np.std(delta_hll, ddof=1)) if len(delta_hll) > 1 else 0.0,
            "delta_hll_min": float(np.min(delta_hll)),
            "delta_hll_max": float(np.max(delta_hll)),
            "delta_avg_hll_mean": float(np.mean(delta_avg_hll)),
            "delta_avg_hll_std": float(np.std(delta_avg_hll, ddof=1)) if len(delta_avg_hll) > 1 else 0.0,
            "delta_bic_mean": float(np.mean(delta_bic)),
            "delta_aic_mean": float(np.mean(delta_aic)),
            "delta_heldout_bic_mean": float(np.mean(delta_heldout_bic)),
            "delta_heldout_aic_mean": float(np.mean(delta_heldout_aic)),
            "cross_cov_norm_weighted_mean": float(np.mean(cross_weighted)),
            "cross_cov_norm_weighted_median": float(np.median(cross_weighted)),
            "cross_cov_norm_max_mean": float(np.mean(cross_max)),
            "best_restart_by_cyl_bic": int(best_run["restart"]),
            "best_seed_by_cyl_bic": int(best_run["seed"]),
            "best_delta_hll": float(best_run["delta_hll"]),
            "best_delta_bic": float(best_run["delta_bic"]),
            "best_delta_aic": float(best_run["delta_aic"]),
            "best_cross_cov_norm_weighted": float(best_run["cross_cov_norm_weighted"]),
        }
        summaries.append(summary)
    return summaries


def cylindrical_mixture_builder(
        d_gauss: int,
        d_vmf: int,
        n_components: int,
        init: str,
        rng: np.random.RandomState,
        ) -> clvm.MixtureModel:
    """Build a cylindrical mixture with ``n_components`` Cylindrical components."""
    d_gauss = _validate_positive_int(d_gauss, name="d_gauss", minimum=1)
    d_vmf = _validate_positive_int(d_vmf, name="d_vmf", minimum=2)
    n_components = _validate_positive_int(n_components, name="n_components", minimum=1)
    return clvm.MixtureModel(
        [clvm.Cylindrical(d_gauss=d_gauss, d_vmf=d_vmf) for _ in range(n_components)],
        init=init,
        rng=rng,
    )


def ind_cylindrical_mixture_builder(
        d_gauss: int,
        d_vmf: int,
        n_components: int,
        init: str,
        rng: np.random.RandomState,
        ) -> clvm.MixtureModel:
    """Build an independent cylindrical mixture with zero within-component cross-dependence."""
    d_gauss = _validate_positive_int(d_gauss, name="d_gauss", minimum=1)
    d_vmf = _validate_positive_int(d_vmf, name="d_vmf", minimum=2)
    n_components = _validate_positive_int(n_components, name="n_components", minimum=1)
    return clvm.MixtureModel(
        [clvm.IndCylindrical(d_gauss=d_gauss, d_vmf=d_vmf) for _ in range(n_components)],
        init=init,
        rng=rng,
    )


def _cylindrical_cross_covariance_summary(model) -> dict[str, Any]:
    """Summarize fitted cross-covariance matrices for a cylindrical mixture."""
    if not hasattr(model, "components") or not hasattr(model, "n_components"):
        raise ValueError("model must be a cylindrical MixtureModel-like object.")
    components = list(model.components)
    if not components:
        raise ValueError("model must contain at least one component.")
    n_components = int(model.n_components)
    if n_components != len(components):
        raise ValueError(
            "model.n_components must match the number of components; "
            f"got {n_components} and {len(components)}."
        )

    matrices = []
    for idx, component in enumerate(components):
        if not hasattr(component, "cross_cov"):
            raise ValueError(f"component {idx} does not expose cross_cov.")
        matrix = np.asarray(component.cross_cov, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"component {idx} cross_cov must be a matrix.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"component {idx} cross_cov contains non-finite values.")
        matrices.append(matrix)

    weights = _mixture_weights(model, n_components=n_components)
    norms = np.asarray([float(np.linalg.norm(matrix, ord="fro")) for matrix in matrices])
    return {
        "cross_cov_matrices": [matrix.copy() for matrix in matrices],
        "cross_cov_norms": norms.tolist(),
        "cross_cov_norm_mean": float(np.mean(norms)),
        "cross_cov_norm_median": float(np.median(norms)),
        "cross_cov_norm_max": float(np.max(norms)),
        "cross_cov_norm_weighted": float(np.sum(weights * norms)),
        "weights": weights.tolist(),
    }


def two_layer_gaussian_vmf_builder(
        d_gauss: int,
        d_vmf: int,
        n_layer1_components: int,
        n_layer2_components: int,
        init_layer1: str,
        init_layer2: str,
        rng: np.random.RandomState,
        ) -> clvm.TwoLayerMoM:
    """Build a two-layer Gaussian/vMF mixture-of-mixtures model."""
    d_gauss = _validate_positive_int(d_gauss, name="d_gauss", minimum=1)
    d_vmf = _validate_positive_int(d_vmf, name="d_vmf", minimum=2)
    n_layer1_components = _validate_positive_int(
        n_layer1_components,
        name="n_layer1_components",
        minimum=1,
    )
    n_layer2_components = _validate_positive_int(
        n_layer2_components,
        name="n_layer2_components",
        minimum=1,
    )

    layer1_mixture = clvm.MixtureModel(
        [clvm.MultivariateGaussian(d_gauss) for _ in range(n_layer1_components)],
        init=init_layer1,
        rng=rng,
    )
    layer2_mixtures = [
        clvm.MixtureModel(
            [clvm.VonMisesFisher(d_vmf) for _ in range(n_layer2_components)],
            init=init_layer2,
            rng=rng,
        )
        for _ in range(n_layer1_components)
    ]
    return clvm.TwoLayerMoM(
        layer1_mixture=layer1_mixture,
        layer2_mixtures=layer2_mixtures,
    )


def two_layer_vmf_gaussian_builder(
        d_vmf: int,
        d_gauss: int,
        n_layer1_components: int,
        n_layer2_components: int,
        init_layer1: str,
        init_layer2: str,
        rng: np.random.RandomState,
        ) -> clvm.TwoLayerMoM:
    """Build a two-layer vMF/Gaussian mixture-of-mixtures model."""
    d_vmf = _validate_positive_int(d_vmf, name="d_vmf", minimum=2)
    d_gauss = _validate_positive_int(d_gauss, name="d_gauss", minimum=1)
    n_layer1_components = _validate_positive_int(
        n_layer1_components,
        name="n_layer1_components",
        minimum=1,
    )
    n_layer2_components = _validate_positive_int(
        n_layer2_components,
        name="n_layer2_components",
        minimum=1,
    )

    layer1_mixture = clvm.MixtureModel(
        [clvm.VonMisesFisher(d_vmf) for _ in range(n_layer1_components)],
        init=init_layer1,
        rng=rng,
    )
    layer2_mixtures = [
        clvm.MixtureModel(
            [clvm.MultivariateGaussian(d_gauss) for _ in range(n_layer2_components)],
            init=init_layer2,
            rng=rng,
        )
        for _ in range(n_layer1_components)
    ]
    return clvm.TwoLayerMoM(
        layer1_mixture=layer1_mixture,
        layer2_mixtures=layer2_mixtures,
    )


def fit_best_cylindrical_mixture(
        x_train,
        x_test,
        *,
        d_gauss: int,
        d_vmf: Optional[int] = None,
        n_components: int,
        family: str = "full",
        n_starts: int = 20,
        start_seeds: Optional[Sequence[int]] = None,
        init: str = "k-means",
        model_builder: Optional[
            Callable[[int, str, np.random.RandomState], clvm.MixtureModel]
        ] = None,
        tol: float = 1e-4,
        max_iter: int = 1000,
        c_step_bool: bool = False,
        verbose: bool = False,
        show_progress: bool = True,
        random_state: Optional[int | np.random.RandomState] = 42,
        fail_fast: bool = False,
        ) -> dict[str, Any]:
    """
    Fit one fixed cylindrical-mixture candidate from several EM starts.

    The retained model is the successful start with the lowest training BIC.
    For a fixed model family and ``K``, that selection is equivalent to
    retaining the maximum training log-likelihood start.
    """
    x_train, x_test, d_gauss, d_vmf = _validate_cylindrical_data(
        x_train,
        x_test,
        d_gauss=d_gauss,
        d_vmf=d_vmf,
    )
    n_components = _validate_positive_int(n_components, name="n_components", minimum=1)
    family = str(family).lower()
    if family not in {"full", "independent"}:
        raise ValueError("family must be 'full' or 'independent'.")

    if model_builder is None:
        def builder(
                k: int,
                init_method: str,
                rng: np.random.RandomState,
                ) -> clvm.MixtureModel:
            if family == "independent":
                return ind_cylindrical_mixture_builder(
                    d_gauss=d_gauss,
                    d_vmf=d_vmf,
                    n_components=k,
                    init=init_method,
                    rng=rng,
                )
            return cylindrical_mixture_builder(
                d_gauss=d_gauss,
                d_vmf=d_vmf,
                n_components=k,
                init=init_method,
                rng=rng,
            )
    else:
        builder = model_builder
    if not callable(builder):
        raise TypeError("model_builder must be callable.")

    seeds = _resolve_start_seeds(
        n_starts=n_starts,
        random_state=random_state,
        start_seeds=start_seeds,
    )
    results: list[dict[str, Any]] = []
    best_model = None
    best_result = None
    descriptor = f"{family} cylindrical K={n_components}"

    for em_start, seed in enumerate(seeds):
        rec: dict[str, Any] = {
            "model_type": "cylindrical_mixture",
            "model_family": family,
            "k": int(n_components),
            "n_components": int(n_components),
            "em_start": int(em_start),
            "em_seed": int(seed),
            "success": False,
            "train_bic": np.inf,
            "train_aic": np.inf,
            "heldout_bic": np.inf,
            "heldout_aic": np.inf,
            "train_log_likelihood": -np.inf,
            "heldout_log_likelihood": -np.inf,
            "train_avg_log_likelihood": -np.inf,
            "heldout_avg_log_likelihood": -np.inf,
            "train_gmpd": 0.0,
            "heldout_gmpd": 0.0,
            "n_free_params": None,
            "n_iter": None,
            "converged": None,
            "fit_time": np.nan,
        }
        try:
            model = builder(n_components, init, np.random.RandomState(seed))
            model_n_components = _safe_getattr(model, "n_components")
            if model_n_components is not None and int(model_n_components) != n_components:
                raise ValueError(
                    "model_builder returned a model with inconsistent n_components: "
                    f"expected {n_components}, got {model_n_components}."
                )

            start_time = time.perf_counter()
            model.fit(
                x_train,
                tol=tol,
                max_iter=max_iter,
                verbose=verbose,
                c_step_bool=c_step_bool,
            )
            fit_time = time.perf_counter() - start_time

            rec.update(_compute_fitted_model_metrics(
                model,
                train_args=(x_train,),
                test_args=(x_test,),
                max_iter=max_iter,
                fit_time=fit_time,
            ))
            rec["success"] = True
            if best_result is None or float(rec["train_bic"]) < float(best_result["train_bic"]):
                best_model = model
                best_result = rec.copy()
        except Exception as exc:  # pragma: no cover - robust notebook runs.
            rec["error"] = repr(exc)
            if fail_fast:
                raise
        finally:
            results.append(rec)
            if show_progress:
                _print_multistart_progress(
                    completed_runs=len(results),
                    total_runs=len(seeds),
                    rec=rec,
                    descriptor=descriptor,
                )

    if best_model is None or best_result is None:
        raise RuntimeError(f"All EM starts failed for {descriptor}.")

    return {
        "best_model": best_model,
        "best_result": best_result,
        "best_bic": float(best_result["train_bic"]),
        "best_k": int(best_result["k"]),
        "best_parameter_summary": best_result.get("parameter_summary"),
        "results": results,
        "model_type": "cylindrical_mixture",
        "model_family": family,
        "k": int(n_components),
        "n_components": int(n_components),
        "n_starts": len(seeds),
        "start_seeds": seeds,
        "selection_metric": "train_bic",
    }


def fit_best_two_layer_mixture(
        layer1_train,
        layer2_train,
        layer1_test,
        layer2_test,
        *,
        family: str,
        n_layer1_components: int,
        n_layer2_components: int,
        n_starts: int = 20,
        start_seeds: Optional[Sequence[int]] = None,
        init_layer1: str = "k-means",
        init_layer2: str = "k-means",
        model_builder: Optional[
            Callable[[int, int, str, str, np.random.RandomState], clvm.TwoLayerMoM]
        ] = None,
        tol: float = 1e-4,
        max_iter: int = 1000,
        c_step_bool: bool = False,
        verbose: bool = False,
        show_progress: bool = True,
        random_state: Optional[int | np.random.RandomState] = 42,
        fail_fast: bool = False,
        ) -> dict[str, Any]:
    """
    Fit one fixed two-layer mixture candidate from several EM starts.

    ``family='gaussian_vmf'`` means Gaussian layer 1 and vMF layer 2.
    ``family='vmf_gaussian'`` means vMF layer 1 and Gaussian layer 2.
    The retained model is the successful start with the lowest training BIC.
    """
    layer1_train = _validate_2d_array(layer1_train, name="layer1_train")
    layer2_train = _validate_2d_array(layer2_train, name="layer2_train")
    layer1_test = _validate_2d_array(layer1_test, name="layer1_test")
    layer2_test = _validate_2d_array(layer2_test, name="layer2_test")
    if layer1_train.shape[0] != layer2_train.shape[0]:
        raise ValueError("layer1_train and layer2_train must have the same number of samples.")
    if layer1_test.shape[0] != layer2_test.shape[0]:
        raise ValueError("layer1_test and layer2_test must have the same number of samples.")
    if layer1_train.shape[1] != layer1_test.shape[1]:
        raise ValueError("layer1_train and layer1_test must have the same number of features.")
    if layer2_train.shape[1] != layer2_test.shape[1]:
        raise ValueError("layer2_train and layer2_test must have the same number of features.")

    family = str(family).lower()
    if family not in {"gaussian_vmf", "vmf_gaussian"}:
        raise ValueError("family must be 'gaussian_vmf' or 'vmf_gaussian'.")
    if family == "gaussian_vmf" and layer2_train.shape[1] < 2:
        raise ValueError("layer2_train must have at least two columns for vMF features.")
    if family == "vmf_gaussian" and layer1_train.shape[1] < 2:
        raise ValueError("layer1_train must have at least two columns for vMF features.")

    n_layer1_components = _validate_positive_int(
        n_layer1_components,
        name="n_layer1_components",
        minimum=1,
    )
    n_layer2_components = _validate_positive_int(
        n_layer2_components,
        name="n_layer2_components",
        minimum=1,
    )

    d_layer1 = int(layer1_train.shape[1])
    d_layer2 = int(layer2_train.shape[1])
    if model_builder is None:
        def builder(
                k: int,
                l_value: int,
                first_init: str,
                second_init: str,
                rng: np.random.RandomState,
                ) -> clvm.TwoLayerMoM:
            if family == "gaussian_vmf":
                return two_layer_gaussian_vmf_builder(
                    d_gauss=d_layer1,
                    d_vmf=d_layer2,
                    n_layer1_components=k,
                    n_layer2_components=l_value,
                    init_layer1=first_init,
                    init_layer2=second_init,
                    rng=rng,
                )
            return two_layer_vmf_gaussian_builder(
                d_vmf=d_layer1,
                d_gauss=d_layer2,
                n_layer1_components=k,
                n_layer2_components=l_value,
                init_layer1=first_init,
                init_layer2=second_init,
                rng=rng,
            )
    else:
        builder = model_builder
    if not callable(builder):
        raise TypeError("model_builder must be callable.")

    seeds = _resolve_start_seeds(
        n_starts=n_starts,
        random_state=random_state,
        start_seeds=start_seeds,
    )
    results: list[dict[str, Any]] = []
    best_model = None
    best_result = None
    descriptor = f"{family} two-layer K={n_layer1_components} L={n_layer2_components}"

    for em_start, seed in enumerate(seeds):
        rec: dict[str, Any] = {
            "model_type": "two_layer_mixture",
            "model_family": family,
            "k": int(n_layer1_components),
            "l": int(n_layer2_components),
            "n_layer1_components": int(n_layer1_components),
            "n_layer2_components": int(n_layer2_components),
            "em_start": int(em_start),
            "em_seed": int(seed),
            "success": False,
            "train_bic": np.inf,
            "train_aic": np.inf,
            "heldout_bic": np.inf,
            "heldout_aic": np.inf,
            "train_log_likelihood": -np.inf,
            "heldout_log_likelihood": -np.inf,
            "train_avg_log_likelihood": -np.inf,
            "heldout_avg_log_likelihood": -np.inf,
            "train_gmpd": 0.0,
            "heldout_gmpd": 0.0,
            "n_free_params": None,
            "n_iter": None,
            "converged": None,
            "fit_time": np.nan,
        }
        try:
            model = builder(
                n_layer1_components,
                n_layer2_components,
                init_layer1,
                init_layer2,
                np.random.RandomState(seed),
            )
            model_k = _safe_getattr(model, "n_layer1_components")
            if model_k is not None and int(model_k) != n_layer1_components:
                raise ValueError(
                    "model_builder returned a model with inconsistent n_layer1_components: "
                    f"expected {n_layer1_components}, got {model_k}."
                )
            layer2_mixtures = _safe_getattr(model, "layer2_mixtures")
            if layer2_mixtures is not None and not all(
                    mixture.n_components == n_layer2_components
                    for mixture in layer2_mixtures
                    ):
                raise ValueError(
                    "model_builder returned a model with inconsistent layer-2 component counts."
                )

            start_time = time.perf_counter()
            model.fit(
                layer1_train,
                layer2_train,
                tol=tol,
                max_iter=max_iter,
                verbose=verbose,
                c_step_bool=c_step_bool,
            )
            fit_time = time.perf_counter() - start_time

            rec.update(_compute_fitted_model_metrics(
                model,
                train_args=(layer1_train, layer2_train),
                test_args=(layer1_test, layer2_test),
                max_iter=max_iter,
                fit_time=fit_time,
            ))
            rec["success"] = True
            if best_result is None or float(rec["train_bic"]) < float(best_result["train_bic"]):
                best_model = model
                best_result = rec.copy()
        except Exception as exc:  # pragma: no cover - robust notebook runs.
            rec["error"] = repr(exc)
            if fail_fast:
                raise
        finally:
            results.append(rec)
            if show_progress:
                _print_multistart_progress(
                    completed_runs=len(results),
                    total_runs=len(seeds),
                    rec=rec,
                    descriptor=descriptor,
                )

    if best_model is None or best_result is None:
        raise RuntimeError(f"All EM starts failed for {descriptor}.")

    return {
        "best_model": best_model,
        "best_result": best_result,
        "best_bic": float(best_result["train_bic"]),
        "best_k": int(best_result["k"]),
        "best_l": int(best_result["l"]),
        "best_parameter_summary": best_result.get("parameter_summary"),
        "results": results,
        "model_type": "two_layer_mixture",
        "model_family": family,
        "k": int(n_layer1_components),
        "l": int(n_layer2_components),
        "n_layer1_components": int(n_layer1_components),
        "n_layer2_components": int(n_layer2_components),
        "n_starts": len(seeds),
        "start_seeds": seeds,
        "selection_metric": "train_bic",
    }


def compare_cylindrical_dependence_by_k(
        x_train,
        x_test,
        *,
        d_gauss: int,
        d_vmf: Optional[int] = None,
        k_grid: Sequence[int] = (1, 2, 3, 4, 5),
        n_restarts: int = 1,
        init: str = "k-means",
        tol: float = 1e-4,
        max_iter: int = 1000,
        c_step_bool: bool = False,
        verbose: bool = False,
        show_progress: bool = True,
        random_state: Optional[int | np.random.RandomState] = 42,
        fail_fast: bool = False,
        ) -> dict[str, Any]:
    """
    Compare full and independent cylindrical mixtures at each fixed ``K``.

    The primary metric is the paired held-out log-likelihood difference
    ``delta_hll = HLL_cyl - HLL_ind``. Positive values favor the full
    cylindrical model. ``delta_bic`` and ``delta_aic`` are computed on the
    training data as ``cyl - ind``; negative values favor the full model.
    """
    x_train, x_test, d_gauss, d_vmf = _validate_cylindrical_data(
        x_train,
        x_test,
        d_gauss=d_gauss,
        d_vmf=d_vmf,
    )
    k_candidates = _validate_component_grid("k_grid", k_grid)
    n_restarts = _validate_n_restarts(n_restarts)

    master_rng = _resolve_rng(random_state)
    results: list[dict[str, Any]] = []
    total_runs = len(k_candidates) * n_restarts
    completed_runs = 0

    for k in k_candidates:
        for restart in range(n_restarts):
            seed = _next_seed(master_rng)
            rec: dict[str, Any] = {
                "k": int(k),
                "n_components": int(k),
                "restart": int(restart),
                "seed": seed,
                "success": False,
                "hll_cyl": -np.inf,
                "hll_ind": -np.inf,
                "delta_hll": -np.inf,
                "avg_hll_cyl": -np.inf,
                "avg_hll_ind": -np.inf,
                "delta_avg_hll": -np.inf,
                "bic_cyl": np.inf,
                "bic_ind": np.inf,
                "delta_bic": np.inf,
                "aic_cyl": np.inf,
                "aic_ind": np.inf,
                "delta_aic": np.inf,
                "heldout_bic_cyl": np.inf,
                "heldout_bic_ind": np.inf,
                "delta_heldout_bic": np.inf,
                "heldout_aic_cyl": np.inf,
                "heldout_aic_ind": np.inf,
                "delta_heldout_aic": np.inf,
                "cross_cov_norm_mean": np.nan,
                "cross_cov_norm_median": np.nan,
                "cross_cov_norm_max": np.nan,
                "cross_cov_norm_weighted": np.nan,
                "fit_time_cyl": np.nan,
                "fit_time_ind": np.nan,
                "n_iter_cyl": None,
                "n_iter_ind": None,
            }
            try:
                cyl_model = cylindrical_mixture_builder(
                    d_gauss=d_gauss,
                    d_vmf=d_vmf,
                    n_components=k,
                    init=init,
                    rng=np.random.RandomState(seed),
                )
                ind_model = ind_cylindrical_mixture_builder(
                    d_gauss=d_gauss,
                    d_vmf=d_vmf,
                    n_components=k,
                    init=init,
                    rng=np.random.RandomState(seed),
                )

                cyl_start = time.perf_counter()
                cyl_model.fit(
                    x_train,
                    tol=tol,
                    max_iter=max_iter,
                    verbose=verbose,
                    c_step_bool=c_step_bool,
                )
                fit_time_cyl = time.perf_counter() - cyl_start

                ind_start = time.perf_counter()
                ind_model.fit(
                    x_train,
                    tol=tol,
                    max_iter=max_iter,
                    verbose=verbose,
                    c_step_bool=c_step_bool,
                )
                fit_time_ind = time.perf_counter() - ind_start

                ll_cyl = np.asarray(cyl_model.log_pdf(x_test), dtype=float)
                ll_ind = np.asarray(ind_model.log_pdf(x_test), dtype=float)
                if ll_cyl.shape != ll_ind.shape:
                    raise ValueError(
                        "model log_pdf outputs have inconsistent shapes: "
                        f"{ll_cyl.shape} and {ll_ind.shape}."
                    )
                if not np.all(np.isfinite(ll_cyl)) or not np.all(np.isfinite(ll_ind)):
                    raise ValueError("held-out log-likelihood contains non-finite values.")

                ll_diff = ll_cyl - ll_ind
                hll_cyl = float(np.sum(ll_cyl))
                hll_ind = float(np.sum(ll_ind))
                delta_hll = float(np.sum(ll_diff))
                cross_summary = _cylindrical_cross_covariance_summary(cyl_model)

                rec.update({
                    "success": True,
                    "hll_cyl": hll_cyl,
                    "hll_ind": hll_ind,
                    "delta_hll": delta_hll,
                    "avg_hll_cyl": float(np.mean(ll_cyl)),
                    "avg_hll_ind": float(np.mean(ll_ind)),
                    "delta_avg_hll": float(np.mean(ll_diff)),
                    "bic_cyl": float(cyl_model.bic_score(x_train)),
                    "bic_ind": float(ind_model.bic_score(x_train)),
                    "aic_cyl": float(cyl_model.aic_score(x_train)),
                    "aic_ind": float(ind_model.aic_score(x_train)),
                    "heldout_bic_cyl": float(cyl_model.bic_score(x_test)),
                    "heldout_bic_ind": float(ind_model.bic_score(x_test)),
                    "heldout_aic_cyl": float(cyl_model.aic_score(x_test)),
                    "heldout_aic_ind": float(ind_model.aic_score(x_test)),
                    "cross_cov_norms": cross_summary["cross_cov_norms"],
                    "cross_cov_norm_mean": cross_summary["cross_cov_norm_mean"],
                    "cross_cov_norm_median": cross_summary["cross_cov_norm_median"],
                    "cross_cov_norm_max": cross_summary["cross_cov_norm_max"],
                    "cross_cov_norm_weighted": cross_summary["cross_cov_norm_weighted"],
                    "fit_time_cyl": float(fit_time_cyl),
                    "fit_time_ind": float(fit_time_ind),
                    "n_iter_cyl": cyl_model.n_iter,
                    "n_iter_ind": ind_model.n_iter,
                })
                rec["delta_bic"] = float(rec["bic_cyl"] - rec["bic_ind"])
                rec["delta_aic"] = float(rec["aic_cyl"] - rec["aic_ind"])
                rec["delta_heldout_bic"] = float(
                    rec["heldout_bic_cyl"] - rec["heldout_bic_ind"]
                )
                rec["delta_heldout_aic"] = float(
                    rec["heldout_aic_cyl"] - rec["heldout_aic_ind"]
                )
            except Exception as exc:  # pragma: no cover - robust notebook search.
                rec["error"] = repr(exc)
                if fail_fast:
                    raise
            finally:
                completed_runs += 1
                results.append(rec)
                if show_progress:
                    _print_dependence_progress(
                        completed_runs=completed_runs,
                        total_runs=total_runs,
                        rec=rec,
                    )

    if not any(rec["success"] for rec in results):
        raise RuntimeError("All paired cylindrical dependence comparison runs failed.")

    results_sorted = sorted(results, key=lambda rec: (int(rec["k"]), int(rec["restart"])))
    summary = _summarize_dependence_results(results)
    return {
        "results": results_sorted,
        "summary": summary,
        "d_gauss": int(d_gauss),
        "d_vmf": int(d_vmf),
        "k_grid": k_candidates,
        "n_restarts": int(n_restarts),
        "metric_notes": {
            "delta_hll": "HLL_cyl - HLL_ind on the held-out set; positive favors full cylindrical.",
            "delta_avg_hll": "Average held-out log-likelihood difference per observation.",
            "delta_bic": "Training BIC_cyl - BIC_ind; negative favors full cylindrical.",
            "delta_aic": "Training AIC_cyl - AIC_ind; negative favors full cylindrical.",
            "delta_heldout_bic": "Held-out BIC_cyl - BIC_ind; negative favors full cylindrical.",
            "delta_heldout_aic": "Held-out AIC_cyl - AIC_ind; negative favors full cylindrical.",
        },
    }


def grid_search_cylindrical_mixture(
        x,
        *,
        d_gauss: int,
        d_vmf: Optional[int] = None,
        k_grid: Sequence[int] = (1, 2, 3, 4, 5),
        n_restarts: int = 1,
        init: str = "k-means",
        model_builder: Optional[
            Callable[[int, str, np.random.RandomState], clvm.MixtureModel]
        ] = None,
        tol: float = 1e-4,
        max_iter: int = 1000,
        c_step_bool: bool = False,
        verbose: bool = False,
        show_progress: bool = True,
        random_state: Optional[int | np.random.RandomState] = 42,
        fail_fast: bool = False,
        ) -> dict[str, Any]:
    """
    Search for the optimal cylindrical-mixture component count by BIC.

    ``x`` must concatenate Gaussian features first and unit-vector vMF features
    second. Lower BIC is better. If ``show_progress`` is true, one line is
    printed after each fit with run progress and BIC.
    """
    x, d_gauss, d_vmf = _validate_cylindrical_feature_array(
        x,
        d_gauss=d_gauss,
        d_vmf=d_vmf,
    )
    k_candidates = _validate_component_grid("k_grid", k_grid)
    n_restarts = _validate_n_restarts(n_restarts)
    if model_builder is None:
        def builder(
                n_components: int,
                init_method: str,
                rng: np.random.RandomState,
                ) -> clvm.MixtureModel:
            return cylindrical_mixture_builder(
                d_gauss=d_gauss,
                d_vmf=d_vmf,
                n_components=n_components,
                init=init_method,
                rng=rng,
            )
    else:
        builder = model_builder
    if not callable(builder):
        raise TypeError("model_builder must be callable.")

    master_rng = _resolve_rng(random_state)
    results: list[dict[str, Any]] = []
    best_model = None
    best_result = None
    total_runs = len(k_candidates) * n_restarts
    completed_runs = 0

    for k in k_candidates:
        for restart in range(n_restarts):
            seed = _next_seed(master_rng)
            rec: dict[str, Any] = {
                "k": int(k),
                "n_components": int(k),
                "restart": int(restart),
                "seed": seed,
                "success": False,
                "bic": np.inf,
                "aic": np.inf,
                "avg_ll": -np.inf,
                "n_iter": None,
                "fit_time": np.nan,
            }
            try:
                model = builder(k, init, np.random.RandomState(seed))
                if model.n_components != k:
                    raise ValueError(
                        "model_builder returned a model with inconsistent n_components: "
                        f"expected {k}, got {model.n_components}."
                    )

                start_time = time.perf_counter()
                model.fit(
                    x,
                    tol=tol,
                    max_iter=max_iter,
                    verbose=verbose,
                    c_step_bool=c_step_bool,
                )
                fit_time = time.perf_counter() - start_time

                bic = float(model.bic_score(x))
                if not np.isfinite(bic):
                    raise ValueError("BIC produced a non-finite value.")

                rec.update({
                    "success": True,
                    "bic": bic,
                    "aic": float(model.aic_score(x)),
                    "avg_ll": float(model.score(x)),
                    "n_iter": model.n_iter,
                    "fit_time": float(fit_time),
                })
                if best_result is None or bic < float(best_result["bic"]):
                    best_model = model
                    best_result = rec.copy()
            except Exception as exc:  # pragma: no cover - robust notebook search.
                rec["error"] = repr(exc)
                if fail_fast:
                    raise
            finally:
                completed_runs += 1
                results.append(rec)
                if show_progress:
                    _print_grid_search_progress(
                        completed_runs=completed_runs,
                        total_runs=total_runs,
                        rec=rec,
                        keys=("k",),
                    )

    if best_model is None or best_result is None:
        raise RuntimeError("All cylindrical mixture grid-search runs failed.")

    results_sorted = sorted(results, key=lambda rec: (not rec["success"], float(rec["bic"])))
    config_results = _summarize_config_results(
        results,
        keys=("k",),
        candidates=[(k,) for k in k_candidates],
    )

    return {
        "best_model": best_model,
        "best_k": int(best_result["k"]),
        "best_bic": float(best_result["bic"]),
        "best_result": best_result,
        "results": results_sorted,
        "config_results": config_results,
    }


def grid_search_ind_cylindrical_mixture(
        x,
        *,
        d_gauss: int,
        d_vmf: Optional[int] = None,
        k_grid: Sequence[int] = (1, 2, 3, 4, 5),
        n_restarts: int = 1,
        init: str = "k-means",
        model_builder: Optional[
            Callable[[int, str, np.random.RandomState], clvm.MixtureModel]
        ] = None,
        tol: float = 1e-4,
        max_iter: int = 1000,
        c_step_bool: bool = False,
        verbose: bool = False,
        show_progress: bool = True,
        random_state: Optional[int | np.random.RandomState] = 42,
        fail_fast: bool = False,
        ) -> dict[str, Any]:
    """
    Search for the optimal independent cylindrical-mixture component count by BIC.

    This has the same return structure as ``grid_search_cylindrical_mixture``,
    but its default builder uses ``IndCylindrical`` components.
    """
    x, d_gauss, d_vmf = _validate_cylindrical_feature_array(
        x,
        d_gauss=d_gauss,
        d_vmf=d_vmf,
    )
    if model_builder is None:
        def builder(
                n_components: int,
                init_method: str,
                rng: np.random.RandomState,
                ) -> clvm.MixtureModel:
            return ind_cylindrical_mixture_builder(
                d_gauss=d_gauss,
                d_vmf=d_vmf,
                n_components=n_components,
                init=init_method,
                rng=rng,
            )
    else:
        builder = model_builder

    return grid_search_cylindrical_mixture(
        x,
        d_gauss=d_gauss,
        d_vmf=d_vmf,
        k_grid=k_grid,
        n_restarts=n_restarts,
        init=init,
        model_builder=builder,
        tol=tol,
        max_iter=max_iter,
        c_step_bool=c_step_bool,
        verbose=verbose,
        show_progress=show_progress,
        random_state=random_state,
        fail_fast=fail_fast,
    )


def grid_search_two_layer_mixture_gaussian_vmf(
        layer1_data,
        layer2_data,
        *,
        k_grid: Sequence[int] = (1, 2, 3, 4, 5),
        l_grid: Sequence[int] = (1, 2, 3),
        n_restarts: int = 1,
        init_layer1: str = "k-means",
        init_layer2: str = "k-means",
        model_builder: Optional[
            Callable[[int, int, str, str, np.random.RandomState], clvm.TwoLayerMoM]
        ] = None,
        tol: float = 1e-4,
        max_iter: int = 1000,
        c_step_bool: bool = False,
        verbose: bool = False,
        show_progress: bool = True,
        random_state: Optional[int | np.random.RandomState] = 42,
        fail_fast: bool = False,
        ) -> dict[str, Any]:
    """
    Search for the optimal ``(K, L)`` in a Gaussian/vMF two-layer mixture.

    ``K`` is the Gaussian layer-1 mixture size. ``L`` is the vMF layer-2
    mixture size attached to each layer-1 component. Lower BIC is better. If
    ``show_progress`` is true, one line is printed after each fit with run
    progress and BIC.
    """
    layer1_data = _validate_2d_array(layer1_data, name="layer1_data")
    layer2_data = _validate_2d_array(layer2_data, name="layer2_data")
    if layer1_data.shape[0] != layer2_data.shape[0]:
        raise ValueError("layer1_data and layer2_data must have the same number of samples.")
    if layer2_data.shape[1] < 2:
        raise ValueError("layer2_data must have at least two columns for vMF features.")

    k_candidates = _validate_component_grid("k_grid", k_grid)
    l_candidates = _validate_component_grid("l_grid", l_grid)
    n_restarts = _validate_n_restarts(n_restarts)
    d_gauss = int(layer1_data.shape[1])
    d_vmf = int(layer2_data.shape[1])

    if model_builder is None:
        def builder(
                n_layer1_components: int,
                n_layer2_components: int,
                first_init: str,
                second_init: str,
                rng: np.random.RandomState,
                ) -> clvm.TwoLayerMoM:
            return two_layer_gaussian_vmf_builder(
                d_gauss=d_gauss,
                d_vmf=d_vmf,
                n_layer1_components=n_layer1_components,
                n_layer2_components=n_layer2_components,
                init_layer1=first_init,
                init_layer2=second_init,
                rng=rng,
            )
    else:
        builder = model_builder
    if not callable(builder):
        raise TypeError("model_builder must be callable.")

    master_rng = _resolve_rng(random_state)
    results: list[dict[str, Any]] = []
    best_model = None
    best_result = None
    total_runs = len(k_candidates) * len(l_candidates) * n_restarts
    completed_runs = 0

    for k in k_candidates:
        for l_value in l_candidates:
            for restart in range(n_restarts):
                seed = _next_seed(master_rng)
                rec: dict[str, Any] = {
                    "k": int(k),
                    "l": int(l_value),
                    "n_layer1_components": int(k),
                    "n_layer2_components": int(l_value),
                    "restart": int(restart),
                    "seed": seed,
                    "success": False,
                    "bic": np.inf,
                    "aic": np.inf,
                    "avg_ll": -np.inf,
                    "n_iter": None,
                    "fit_time": np.nan,
                }
                try:
                    model = builder(
                        k,
                        l_value,
                        init_layer1,
                        init_layer2,
                        np.random.RandomState(seed),
                    )
                    if model.n_layer1_components != k:
                        raise ValueError(
                            "model_builder returned a model with inconsistent n_layer1_components: "
                            f"expected {k}, got {model.n_layer1_components}."
                        )
                    if not all(mix.n_components == l_value for mix in model.layer2_mixtures):
                        raise ValueError(
                            "model_builder returned a model with inconsistent layer-2 component counts."
                        )

                    start_time = time.perf_counter()
                    model.fit(
                        layer1_data,
                        layer2_data,
                        tol=tol,
                        max_iter=max_iter,
                        verbose=verbose,
                        c_step_bool=c_step_bool,
                    )
                    fit_time = time.perf_counter() - start_time

                    bic = float(model.bic_score(layer1_data, layer2_data))
                    if not np.isfinite(bic):
                        raise ValueError("BIC produced a non-finite value.")

                    rec.update({
                        "success": True,
                        "bic": bic,
                        "aic": float(model.aic_score(layer1_data, layer2_data)),
                        "avg_ll": float(model.score(layer1_data, layer2_data)),
                        "n_iter": model.n_iter,
                        "fit_time": float(fit_time),
                    })
                    if best_result is None or bic < float(best_result["bic"]):
                        best_model = model
                        best_result = rec.copy()
                except Exception as exc:  # pragma: no cover - robust notebook search.
                    rec["error"] = repr(exc)
                    if fail_fast:
                        raise
                finally:
                    completed_runs += 1
                    results.append(rec)
                    if show_progress:
                        _print_grid_search_progress(
                            completed_runs=completed_runs,
                            total_runs=total_runs,
                            rec=rec,
                            keys=("k", "l"),
                        )

    if best_model is None or best_result is None:
        raise RuntimeError("All two-layer mixture grid-search runs failed.")

    results_sorted = sorted(results, key=lambda rec: (not rec["success"], float(rec["bic"])))
    config_results = _summarize_config_results(
        results,
        keys=("k", "l"),
        candidates=[(k, l_value) for k in k_candidates for l_value in l_candidates],
    )

    return {
        "best_model": best_model,
        "best_k": int(best_result["k"]),
        "best_l": int(best_result["l"]),
        "best_bic": float(best_result["bic"]),
        "best_result": best_result,
        "results": results_sorted,
        "config_results": config_results,
    }


def grid_search_two_layer_vmf_gaussian(
        vmf_data,
        gaussian_data,
        *,
        k_grid: Sequence[int] = (1, 2, 3, 4, 5),
        l_grid: Sequence[int] = (1, 2, 3),
        n_restarts: int = 1,
        init_layer1: str = "k-means",
        init_layer2: str = "k-means",
        model_builder: Optional[
            Callable[[int, int, str, str, np.random.RandomState], clvm.TwoLayerMoM]
        ] = None,
        tol: float = 1e-4,
        max_iter: int = 1000,
        c_step_bool: bool = False,
        verbose: bool = False,
        show_progress: bool = True,
        random_state: Optional[int | np.random.RandomState] = 42,
        fail_fast: bool = False,
        ) -> dict[str, Any]:
    """
    Search for the optimal ``(K, L)`` in a vMF/Gaussian two-layer mixture.

    ``K`` is the vMF layer-1 mixture size. ``L`` is the Gaussian layer-2
    mixture size attached to each layer-1 component. Lower BIC is better. If
    ``show_progress`` is true, one line is printed after each fit with run
    progress and BIC.
    """
    vmf_data = _validate_2d_array(vmf_data, name="vmf_data")
    gaussian_data = _validate_2d_array(gaussian_data, name="gaussian_data")
    if vmf_data.shape[0] != gaussian_data.shape[0]:
        raise ValueError("vmf_data and gaussian_data must have the same number of samples.")
    if vmf_data.shape[1] < 2:
        raise ValueError("vmf_data must have at least two columns for vMF features.")

    k_candidates = _validate_component_grid("k_grid", k_grid)
    l_candidates = _validate_component_grid("l_grid", l_grid)
    n_restarts = _validate_n_restarts(n_restarts)
    d_vmf = int(vmf_data.shape[1])
    d_gauss = int(gaussian_data.shape[1])

    if model_builder is None:
        def builder(
                n_layer1_components: int,
                n_layer2_components: int,
                first_init: str,
                second_init: str,
                rng: np.random.RandomState,
                ) -> clvm.TwoLayerMoM:
            return two_layer_vmf_gaussian_builder(
                d_vmf=d_vmf,
                d_gauss=d_gauss,
                n_layer1_components=n_layer1_components,
                n_layer2_components=n_layer2_components,
                init_layer1=first_init,
                init_layer2=second_init,
                rng=rng,
            )
    else:
        builder = model_builder
    if not callable(builder):
        raise TypeError("model_builder must be callable.")

    master_rng = _resolve_rng(random_state)
    results: list[dict[str, Any]] = []
    best_model = None
    best_result = None
    total_runs = len(k_candidates) * len(l_candidates) * n_restarts
    completed_runs = 0

    for k in k_candidates:
        for l_value in l_candidates:
            for restart in range(n_restarts):
                seed = _next_seed(master_rng)
                rec: dict[str, Any] = {
                    "k": int(k),
                    "l": int(l_value),
                    "n_layer1_components": int(k),
                    "n_layer2_components": int(l_value),
                    "restart": int(restart),
                    "seed": seed,
                    "success": False,
                    "bic": np.inf,
                    "aic": np.inf,
                    "avg_ll": -np.inf,
                    "n_iter": None,
                    "fit_time": np.nan,
                }
                try:
                    model = builder(
                        k,
                        l_value,
                        init_layer1,
                        init_layer2,
                        np.random.RandomState(seed),
                    )
                    if model.n_layer1_components != k:
                        raise ValueError(
                            "model_builder returned a model with inconsistent n_layer1_components: "
                            f"expected {k}, got {model.n_layer1_components}."
                        )
                    if not all(mix.n_components == l_value for mix in model.layer2_mixtures):
                        raise ValueError(
                            "model_builder returned a model with inconsistent layer-2 component counts."
                        )

                    start_time = time.perf_counter()
                    model.fit(
                        vmf_data,
                        gaussian_data,
                        tol=tol,
                        max_iter=max_iter,
                        verbose=verbose,
                        c_step_bool=c_step_bool,
                    )
                    fit_time = time.perf_counter() - start_time

                    bic = float(model.bic_score(vmf_data, gaussian_data))
                    if not np.isfinite(bic):
                        raise ValueError("BIC produced a non-finite value.")

                    rec.update({
                        "success": True,
                        "bic": bic,
                        "aic": float(model.aic_score(vmf_data, gaussian_data)),
                        "avg_ll": float(model.score(vmf_data, gaussian_data)),
                        "n_iter": model.n_iter,
                        "fit_time": float(fit_time),
                    })
                    if best_result is None or bic < float(best_result["bic"]):
                        best_model = model
                        best_result = rec.copy()
                except Exception as exc:  # pragma: no cover - robust notebook search.
                    rec["error"] = repr(exc)
                    if fail_fast:
                        raise
                finally:
                    completed_runs += 1
                    results.append(rec)
                    if show_progress:
                        _print_grid_search_progress(
                            completed_runs=completed_runs,
                            total_runs=total_runs,
                            rec=rec,
                            keys=("k", "l"),
                        )

    if best_model is None or best_result is None:
        raise RuntimeError("All two-layer vMF/Gaussian grid-search runs failed.")

    results_sorted = sorted(results, key=lambda rec: (not rec["success"], float(rec["bic"])))
    config_results = _summarize_config_results(
        results,
        keys=("k", "l"),
        candidates=[(k, l_value) for k in k_candidates for l_value in l_candidates],
    )

    return {
        "best_model": best_model,
        "best_k": int(best_result["k"]),
        "best_l": int(best_result["l"]),
        "best_bic": float(best_result["bic"]),
        "best_result": best_result,
        "results": results_sorted,
        "config_results": config_results,
    }


__all__ = [
    "compare_cylindrical_dependence_by_k",
    "cylindrical_mixture_builder",
    "fit_best_cylindrical_mixture",
    "fit_best_two_layer_mixture",
    "grid_search_cylindrical_mixture",
    "grid_search_ind_cylindrical_mixture",
    "grid_search_two_layer_mixture_gaussian_vmf",
    "grid_search_two_layer_vmf_gaussian",
    "ind_cylindrical_mixture_builder",
    "two_layer_gaussian_vmf_builder",
    "two_layer_vmf_gaussian_builder",
]
