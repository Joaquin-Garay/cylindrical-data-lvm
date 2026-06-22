"""Calibration helpers for model selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np

from ..core.types import Array

if TYPE_CHECKING:
    from ..mixtures.mixture import MixtureModel
    from ..hierarchical import TwoLayerMoM


def _validate_component_grid(name: str, values: Sequence[int]) -> list[int]:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one value.")

    out: list[int] = []
    for v in values:
        if not isinstance(v, (int, np.integer)) or int(v) < 1:
            raise ValueError(f"All values in {name} must be integers >= 1.")
        out.append(int(v))
    return out


def _validate_selection(selection: str) -> str:
    if selection not in {"best", "mean", "median", "mean_plus_2std"}:
        raise ValueError(
            "selection must be one of {'best', 'mean', 'median', 'mean_plus_2std'}; "
        )
    return selection


def _validate_score_metric(score_metric: str) -> str:
    if score_metric not in {"nll", "bic"}:
        raise ValueError("score_metric must be one of {'nll', 'bic'}.")
    return score_metric


def _score_metric_higher_is_better(score_metric: str) -> bool:
    # Current built-in metrics are loss-like (lower is better).
    metric_direction = {
        "nll": False,
        "bic": False,
    }
    return metric_direction[score_metric]


def _selection_higher_is_better(selection: str, *, metric_higher_is_better: bool) -> bool:
    # Risk-averse aggregate is explicitly lower-is-better.
    if selection == "mean_plus_2std":
        return False
    return metric_higher_is_better


def _validate_cv(cv: int, n_obs: int) -> int:
    if not isinstance(cv, (int, np.integer)) or int(cv) < 1:
        raise ValueError("cv must be an integer >= 1.")
    cv = int(cv)
    if cv > int(n_obs):
        raise ValueError(f"cv cannot exceed number of samples ({n_obs}).")
    return cv


def _build_cv_splits(
        n_obs: int,
        *,
        cv: int,
        shuffle: bool,
        rng: np.random.RandomState,
        ) -> list[tuple[Array, Array]]:
    indices = np.arange(n_obs, dtype=int)
    if shuffle:
        indices = rng.permutation(indices)
    if cv == 1:
        return [(indices, indices)]

    folds = np.array_split(indices, cv)
    splits: list[tuple[Array, Array]] = []
    for fold_idx in range(cv):
        val_idx = folds[fold_idx]
        if val_idx.size == 0:
            continue
        train_parts = [folds[j] for j in range(cv) if j != fold_idx]
        train_idx = np.concatenate(train_parts, axis=0) if train_parts else np.empty(0, dtype=int)
        splits.append((train_idx, val_idx))

    if len(splits) < 1:
        raise RuntimeError("Unable to build at least one non-empty CV fold.")
    return splits


def _selection_score(values: Array, selection: str, *, higher_is_better: bool) -> float:
    if selection == "best":
        return float(np.max(values) if higher_is_better else np.min(values))
    if selection == "mean_plus_2std":
        return float(np.mean(values) + 2.0 * np.std(values))
    if selection == "median":
        return float(np.median(values))
    return float(np.mean(values))


def _select_representative_run(
        runs: list[dict[str, Any]],
        *,
        selection: str,
        selection_score: float,
        metric_higher_is_better: bool,
        selection_higher_is_better: bool,
        metric_key: str = "bic",
        ) -> dict[str, Any]:
    if selection == "best":
        if metric_higher_is_better:
            return max(runs, key=lambda r: float(r[metric_key]))
        return min(runs, key=lambda r: float(r[metric_key]))
    if selection_higher_is_better:
        return min(
            runs,
            key=lambda r: (abs(float(r[metric_key]) - selection_score), -float(r[metric_key])),
        )
    return min(
        runs,
        key=lambda r: (abs(float(r[metric_key]) - selection_score), float(r[metric_key])),
    )


def _default_two_layer_mom_builder(
        n_layer1_components: int,
        n_layer2_components: int,
        init_layer1: str,
        init_layer2: str,
        rng: np.random.RandomState,
        ) -> "TwoLayerMoM":
    """
    Build a default soccer MoM model: Gaussian(2D) in layer 1 and VonMises in layer 2.
    """
    # Local imports avoid package import cycles at module import time.
    from ..distributions import MultivariateGaussian, VonMises
    from ..hierarchical import TwoLayerMoM
    from ..mixtures.mixture import MixtureModel

    layer1_mixture = MixtureModel(
        [MultivariateGaussian(2) for _ in range(n_layer1_components)],
        init=init_layer1,
        rng=rng,
    )
    layer2_mixtures = [
        MixtureModel(
            [VonMises() for _ in range(n_layer2_components)],
            init=init_layer2,
            rng=rng,
        )
        for _ in range(n_layer1_components)
    ]
    return TwoLayerMoM(layer1_mixture=layer1_mixture, layer2_mixtures=layer2_mixtures)


def calibrate_mixture_by_cv_grid_search(
        x,
        *,
        n_components_grid: Sequence[int] = (2, 3, 4),
        cv: int = 5,
        n_restarts: int = 2,
        init: str = "k-means",
        model_builder: Optional[
            Callable[[int, str, np.random.RandomState], "MixtureModel"]
        ] = None,
        tol: float = 1e-4,
        max_iter: int = 300,
        m_step_case: str = "bregman",
        c_step_bool: bool = False,
        verbose: bool = False,
        random_state: Optional[int] = 42,
        fail_fast: bool = False,
        selection: str = "mean",
        score_metric: str = "nll",
        shuffle: bool = True,
        ) -> dict[str, Any]:
    """
    Calibrate a mixture model with a CV grid-search and restart aggregation.

    For each ``n_components`` candidate, the function evaluates folds first,
    then restarts inside each fold. The per-run validation score is controlled
    by ``score_metric``:

    - ``"nll"``: ``-sum(log_pdf(x_val))``
    - ``"bic"``: ``bic_score(x_val)``

    For currently supported metrics (``"nll"``, ``"bic"``), lower is better.
    For each configuration, all successful restart scores across all folds are
    pooled into a single vector, and ``selection`` (``"best"``, ``"mean"``,
    ``"median"``, or ``"mean_plus_2std"``) is applied on that pooled set.

    Parameters
    ----------
    x : array-like, shape (n_obs,) or (n_obs, d)
        Input observations.
    n_components_grid : sequence of int, default=(2, 3, 4)
        Candidate numbers of mixture components.
    cv : int, default=5
        Number of folds. ``cv=1`` is supported and means a single full-data
        fold (train=validation=data).
    n_restarts : int, default=2
        Number of random initializations per fold and configuration.
    init : {"k-means", "random"}, default="k-means"
        Initialization method passed to the mixture model.
    model_builder : callable or None, default=None
        Builder receiving ``(n_components, init, rng)`` and returning a fresh
        ``MixtureModel``. If ``None``, a Gaussian default is used.
    tol, max_iter, m_step_case, c_step_bool, verbose :
        Training options forwarded to ``MixtureModel.fit``.
    random_state : int or None, default=42
        Master seed controlling fold shuffle and all restart seeds.
    fail_fast : bool, default=False
        If ``True``, raise immediately on the first failed run; otherwise keep
        searching and mark failed runs/configurations.
    selection : {"best", "mean", "median", "mean_plus_2std"}, default="mean"
        Aggregator applied to pooled successful restart scores from all folds.
        ``"best"`` follows score direction: min for lower-is-better metrics and
        max for higher-is-better metrics. ``"max"`` is accepted as a
        backward-compatible alias of ``"best"``.
        ``"mean_plus_2std"`` computes ``mean(scores) + 2*std(scores)`` and is
        always treated as lower-is-better.
    score_metric : {"nll", "bic"}, default="nll"
        Validation metric used per run.
    shuffle : bool, default=True
        Whether to shuffle sample indices before fold splitting.

    Returns
    -------
    dict
        Search summary with keys:

        - ``best_model``: refit model on full data for the selected config.
        - ``best_cv_score``: selected configuration score
          (same as ``best_selection_score``).
        - ``best_bic_refit``: BIC of ``best_model`` on full data.
        - ``best_selection_score``: selection score on pooled fold+restart scores.
        - ``best_config``: selected hyperparameters and run metadata.
        - ``selection``, ``score_metric``, ``cv``, ``shuffle``.
        - ``results``: per-(config, fold, restart) run records.
        - ``config_results``: per-configuration aggregated summaries.

    Notes
    -----
    A configuration is considered successful only if every fold has at least
    one successful restart.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim not in {1, 2}:
        raise ValueError("x must be a 1D or 2D array.")
    if x.shape[0] < 1:
        raise ValueError("x must contain at least one sample.")
    if not isinstance(n_restarts, (int, np.integer)) or int(n_restarts) < 1:
        raise ValueError("n_restarts must be an integer >= 1.")
    n_restarts = int(n_restarts)
    selection = _validate_selection(selection)
    score_metric = _validate_score_metric(score_metric)
    metric_higher_is_better = _score_metric_higher_is_better(score_metric)
    selection_higher_is_better = _selection_higher_is_better(
        selection,
        metric_higher_is_better=metric_higher_is_better,
    )
    failed_cv_score = -np.inf if metric_higher_is_better else np.inf
    failed_selection_score = -np.inf if selection_higher_is_better else np.inf
    cv = _validate_cv(cv, x.shape[0])

    n_candidates = _validate_component_grid("n_components_grid", n_components_grid)

    if model_builder is None:
        from ..distributions import MultivariateGaussian, UnivariateGaussian
        from ..mixtures.mixture import MixtureModel

        def default_builder(
                n_components: int,
                init_method: str,
                rng: np.random.RandomState,
        ) -> MixtureModel:
            if x.ndim == 1:
                components = [UnivariateGaussian() for _ in range(n_components)]
            else:
                components = [MultivariateGaussian(x.shape[1]) for _ in range(n_components)]
            return MixtureModel(components=components, init=init_method, rng=rng)

        builder = default_builder
    else:
        builder = model_builder

    if not callable(builder):
        raise TypeError("model_builder must be callable.")

    master_rng = np.random.RandomState(random_state)
    cv_splits = _build_cv_splits(
        x.shape[0],
        cv=cv,
        shuffle=bool(shuffle),
        rng=master_rng,
    )
    n_folds = len(cv_splits)
    results: list[dict[str, Any]] = []
    successful_runs_by_config: dict[tuple[int], list[dict[str, Any]]] = {}
    successful_runs_by_config_and_fold: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for n_components in n_candidates:
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            for restart in range(n_restarts):
                seed = int(master_rng.randint(np.iinfo(np.int32).max))
                run_rng = np.random.RandomState(seed)
                rec: dict[str, Any] = {
                    "n_components": n_components,
                    "fold": int(fold_idx),
                    "restart": restart,
                    "seed": seed,
                    "cv": int(n_folds),
                    "shuffle": bool(shuffle),
                    "score_metric": score_metric,
                }

                try:
                    model = builder(n_components, init, run_rng)
                    if model.n_components != n_components:
                        raise ValueError(
                            "model_builder returned a model with inconsistent n_components: "
                            f"expected {n_components}, got {model.n_components}."
                        )

                    model.fit(
                        x[train_idx],
                        tol=tol,
                        max_iter=max_iter,
                        verbose=verbose,
                        m_step_case=m_step_case,
                        c_step_bool=c_step_bool,
                    )
                    ll_val = np.asarray(model.log_pdf(x[val_idx]), dtype=float)
                    if not np.all(np.isfinite(ll_val)):
                        raise ValueError("Validation log-likelihood produced non-finite values.")

                    fold_nll = float(-np.sum(ll_val))
                    if score_metric == "nll":
                        fold_score = fold_nll
                        fold_bic = np.nan
                    else:
                        fold_bic = float(model.bic_score(x[val_idx]))
                        if not np.isfinite(fold_bic):
                            raise ValueError("Validation BIC produced non-finite values.")
                        fold_score = fold_bic

                    n_iter = model.n_iter
                    rec.update({
                        "success": True,
                        "cv_score": fold_score,
                        "fold_nll": fold_nll,
                        "fold_bic": fold_bic,
                        "n_iter": n_iter,
                    })
                    run_info = {
                        "cv_score": fold_score,
                        "fold_nll": fold_nll,
                        "fold_bic": fold_bic,
                        "fold": int(fold_idx),
                        "restart": restart,
                        "seed": seed,
                    }
                    key = (n_components,)
                    successful_runs_by_config.setdefault(key, []).append(run_info)
                    successful_runs_by_config_and_fold.setdefault((n_components, int(fold_idx)), []).append(run_info)
                except Exception as exc:  # pragma: no cover - kept for robust search.
                    rec.update({
                        "success": False,
                        "cv_score": failed_cv_score,
                        "fold_nll": np.inf,
                        "fold_bic": np.inf if score_metric == "bic" else np.nan,
                        "n_iter": None,
                        "error": repr(exc),
                    })
                    if fail_fast:
                        raise

                results.append(rec)

    config_results: list[dict[str, Any]] = []
    for n_components in n_candidates:
        successful_folds = 0
        failed_folds = 0
        successful_restarts_total = 0
        failed_restarts_total = 0
        for fold_idx in range(n_folds):
            runs = successful_runs_by_config_and_fold.get((n_components, fold_idx), [])
            successful_restarts_total += len(runs)
            failed_restarts_total += n_restarts - len(runs)
            if not runs:
                failed_folds += 1
                continue
            successful_folds += 1

        if successful_folds == n_folds:
            all_runs = successful_runs_by_config[(n_components,)]
            scores = np.asarray([float(r["cv_score"]) for r in all_runs], dtype=float)
            score = _selection_score(
                scores,
                selection,
                higher_is_better=metric_higher_is_better,
            )
            summary = {
                "n_components": n_components,
                "successful_restarts": int(successful_restarts_total),
                "failed_restarts": int(failed_restarts_total),
                "successful_folds": int(successful_folds),
                "failed_folds": int(failed_folds),
                "cv_score_mean": float(np.mean(scores)),
                "cv_score_median": float(np.median(scores)),
                "cv_score_std": float(np.std(scores)),
                "cv_score_min": float(np.min(scores)),
                "cv_score_max": float(np.max(scores)),
                "selection_score": float(score),
                "selection": selection,
                "score_metric": score_metric,
                "fold_aggregation": "all_folds_restarts",
                "success": True,
            }
        else:
            summary = {
                "n_components": n_components,
                "successful_restarts": int(successful_restarts_total),
                "failed_restarts": int(failed_restarts_total),
                "successful_folds": int(successful_folds),
                "failed_folds": int(failed_folds),
                "cv_score_mean": np.inf,
                "cv_score_median": np.inf,
                "cv_score_std": np.inf,
                "cv_score_min": np.inf,
                "cv_score_max": np.inf,
                "selection_score": failed_selection_score,
                "selection": selection,
                "score_metric": score_metric,
                "fold_aggregation": "all_folds_restarts",
                "success": False,
            }
        config_results.append(summary)

    successful_configs = [row for row in config_results if row["success"]]
    if not successful_configs:
        raise RuntimeError("All grid-search runs failed. Set fail_fast=True to surface the first error.")

    if selection_higher_is_better:
        best_config_summary = max(successful_configs, key=lambda row: float(row["selection_score"]))
    else:
        best_config_summary = min(successful_configs, key=lambda row: float(row["selection_score"]))
    best_key = (int(best_config_summary["n_components"]),)
    best_runs = successful_runs_by_config[best_key]
    best_run = _select_representative_run(
        best_runs,
        selection=selection,
        selection_score=float(best_config_summary["selection_score"]),
        metric_higher_is_better=metric_higher_is_better,
        selection_higher_is_better=selection_higher_is_better,
        metric_key="cv_score",
    )

    best_seed = int(best_run["seed"])
    best_components = int(best_config_summary["n_components"])
    best_rng = np.random.RandomState(best_seed)
    best_model = builder(best_components, init, best_rng)
    if best_model.n_components != best_components:
        raise ValueError(
            "model_builder returned a model with inconsistent n_components during refit: "
            f"expected {best_components}, got {best_model.n_components}."
        )
    best_model.fit(
        x,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
        m_step_case=m_step_case,
        c_step_bool=c_step_bool,
    )

    best_n_iter = best_model.n_iter
    best_cv_score = float(best_config_summary["selection_score"])
    best_bic_refit = float(best_model.bic_score(x))
    best_config = {
        "n_components": best_components,
        "fold": int(best_run["fold"]),
        "restart": int(best_run["restart"]),
        "seed": best_seed,
        "n_iter": best_n_iter,
        "selection": selection,
        "score_metric": score_metric,
        "fold_aggregation": "all_folds_restarts",
        "selection_score": float(best_config_summary["selection_score"]),
        "cv_score_mean": float(best_config_summary["cv_score_mean"]),
        "cv_score_median": float(best_config_summary["cv_score_median"]),
        "cv_score_std": float(best_config_summary["cv_score_std"]),
        "cv_score_min": float(best_config_summary["cv_score_min"]),
        "cv_score_max": float(best_config_summary["cv_score_max"]),
    }

    if metric_higher_is_better:
        results_sorted = sorted(results, key=lambda row: (not row["success"], -row["cv_score"]))
    else:
        results_sorted = sorted(results, key=lambda row: (not row["success"], row["cv_score"]))
    if selection_higher_is_better:
        config_results_sorted = sorted(config_results, key=lambda row: (not row["success"], -row["selection_score"]))
    else:
        config_results_sorted = sorted(config_results, key=lambda row: (not row["success"], row["selection_score"]))
    return {
        "best_model": best_model,
        "best_cv_score": best_cv_score,
        "best_bic_refit": best_bic_refit,
        "best_selection_score": float(best_config_summary["selection_score"]),
        "best_config": best_config,
        "selection": selection,
        "score_metric": score_metric,
        "cv": int(n_folds),
        "shuffle": bool(shuffle),
        "results": results_sorted,
        "config_results": config_results_sorted,
    }


def calibrate_mom_by_cv_grid_search(
        layer1_data,
        layer2_data,
        *,
        n_layer1_grid: Sequence[int] = (2, 3, 4),
        n_layer2_grid: Sequence[int] = (1, 2),
        cv: int = 5,
        n_restarts: int = 2,
        init_layer1: str = "k-means",
        init_layer2: str = "k-means",
        model_builder: Optional[
            Callable[[int, int, str, str, np.random.RandomState], "TwoLayerMoM"]
        ] = None,
        tol: float = 1e-4,
        max_iter: int = 300,
        m_step_case: str = "classic",
        c_step_bool: bool = False,
        verbose: bool = False,
        random_state: Optional[int] = 42,
        fail_fast: bool = False,
        selection: str = "mean",
        score_metric: str = "nll",
        shuffle: bool = True,
        ) -> dict[str, Any]:
    """
    Calibrate a two-layer MoM with a CV grid-search and restart aggregation.

    For each ``(n_layer1_components, n_layer2_components)`` candidate, the
    function evaluates folds first, then restarts inside each fold. The per-run
    validation score is controlled by ``score_metric``:

    - ``"nll"``: ``-sum(log_pdf(layer1_val, layer2_val))``
    - ``"bic"``: ``bic_score(layer1_val, layer2_val)``

    For currently supported metrics (``"nll"``, ``"bic"``), lower is better.
    For each configuration, all successful restart scores across all folds are
    pooled into a single vector, and ``selection`` (``"best"``, ``"mean"``,
    ``"median"``, or ``"mean_plus_2std"``) is applied on that pooled set.

    Parameters
    ----------
    layer1_data : array-like, shape (n_obs, d1)
        First-layer observations.
    layer2_data : array-like, shape (n_obs, d2)
        Second-layer observations aligned with ``layer1_data``.
    n_layer1_grid : sequence of int, default=(2, 3, 4)
        Candidate layer-1 component counts.
    n_layer2_grid : sequence of int, default=(1, 2)
        Candidate layer-2 component counts per layer-1 component.
    cv : int, default=5
        Number of folds. ``cv=1`` is supported and means a single full-data
        fold (train=validation=data).
    n_restarts : int, default=2
        Number of random initializations per fold and configuration.
    init_layer1, init_layer2 : str, default="k-means"
        Initialization modes passed to the model builder.
    model_builder : callable or None, default=None
        Builder receiving ``(n_layer1, n_layer2, init_layer1, init_layer2, rng)``
        and returning a fresh ``TwoLayerMoM``. If ``None``, a Gaussian+VonMises
        default builder is used.
    tol, max_iter, m_step_case, c_step_bool, verbose :
        Training options forwarded to ``TwoLayerMoM.fit``.
    random_state : int or None, default=42
        Master seed controlling fold shuffle and all restart seeds.
    fail_fast : bool, default=False
        If ``True``, raise immediately on the first failed run; otherwise keep
        searching and mark failed runs/configurations.
    selection : {"best", "mean", "median", "mean_plus_2std"}, default="mean"
        Aggregator applied to pooled successful restart scores from all folds.
        ``"best"`` follows score direction: min for lower-is-better metrics and
        max for higher-is-better metrics. ``"max"`` is accepted as a
        backward-compatible alias of ``"best"``.
        ``"mean_plus_2std"`` computes ``mean(scores) + 2*std(scores)`` and is
        always treated as lower-is-better.
    score_metric : {"nll", "bic"}, default="nll"
        Validation metric used per run.
    shuffle : bool, default=True
        Whether to shuffle sample indices before fold splitting.

    Returns
    -------
    dict
        Search summary with keys:

        - ``best_model``: refit model on full data for the selected config.
        - ``best_cv_score``: selected configuration score
          (same as ``best_selection_score``).
        - ``best_bic_refit``: BIC of ``best_model`` on full data.
        - ``best_selection_score``: selection score on pooled fold+restart scores.
        - ``best_config``: selected hyperparameters and run metadata.
        - ``selection``, ``score_metric``, ``cv``, ``shuffle``.
        - ``results``: per-(config, fold, restart) run records.
        - ``config_results``: per-configuration aggregated summaries.

    Notes
    -----
    A configuration is considered successful only if every fold has at least
    one successful restart.
    """
    layer1_data = np.asarray(layer1_data, dtype=float)
    layer2_data = np.asarray(layer2_data, dtype=float)
    if layer1_data.ndim != 2 or layer2_data.ndim != 2:
        raise ValueError("layer1_data and layer2_data must be 2D arrays.")
    if layer1_data.shape[0] != layer2_data.shape[0]:
        raise ValueError("layer1_data and layer2_data must have the same number of samples.")
    if layer1_data.shape[0] < 1:
        raise ValueError("Need at least one sample.")
    if not isinstance(n_restarts, (int, np.integer)) or int(n_restarts) < 1:
        raise ValueError("n_restarts must be an integer >= 1.")
    n_restarts = int(n_restarts)
    selection = _validate_selection(selection)
    score_metric = _validate_score_metric(score_metric)
    metric_higher_is_better = _score_metric_higher_is_better(score_metric)
    selection_higher_is_better = _selection_higher_is_better(
        selection,
        metric_higher_is_better=metric_higher_is_better,
    )
    failed_cv_score = -np.inf if metric_higher_is_better else np.inf
    failed_selection_score = -np.inf if selection_higher_is_better else np.inf
    cv = _validate_cv(cv, layer1_data.shape[0])

    layer1_candidates = _validate_component_grid("n_layer1_grid", n_layer1_grid)
    layer2_candidates = _validate_component_grid("n_layer2_grid", n_layer2_grid)

    builder = _default_two_layer_mom_builder if model_builder is None else model_builder
    if not callable(builder):
        raise TypeError("model_builder must be callable.")

    master_rng = np.random.RandomState(random_state)
    cv_splits = _build_cv_splits(
        layer1_data.shape[0],
        cv=cv,
        shuffle=bool(shuffle),
        rng=master_rng,
    )
    n_folds = len(cv_splits)
    results: list[dict[str, Any]] = []
    successful_runs_by_config: dict[tuple[int, int], list[dict[str, Any]]] = {}
    successful_runs_by_config_and_fold: dict[tuple[int, int, int], list[dict[str, Any]]] = {}

    for n_layer1 in layer1_candidates:
        for n_layer2 in layer2_candidates:
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                for restart in range(n_restarts):
                    seed = int(master_rng.randint(np.iinfo(np.int32).max))
                    run_rng = np.random.RandomState(seed)
                    rec: dict[str, Any] = {
                        "n_layer1_components": n_layer1,
                        "n_layer2_components": n_layer2,
                        "fold": int(fold_idx),
                        "restart": restart,
                        "seed": seed,
                        "cv": int(n_folds),
                        "shuffle": bool(shuffle),
                        "score_metric": score_metric,
                    }

                    try:
                        model = builder(n_layer1, n_layer2, init_layer1, init_layer2, run_rng)
                        if model.n_layer1_components != n_layer1:
                            raise ValueError(
                                "model_builder returned a model with inconsistent n_layer1_components: "
                                f"expected {n_layer1}, got {model.n_layer1_components}."
                            )
                        if not all(mix.n_components == n_layer2 for mix in model.layer2_mixtures):
                            raise ValueError(
                                "model_builder returned a model with inconsistent layer-2 component counts."
                            )

                        model.fit(
                            layer1_data[train_idx],
                            layer2_data[train_idx],
                            tol=tol,
                            max_iter=max_iter,
                            verbose=verbose,
                            m_step_case=m_step_case,
                            c_step_bool=c_step_bool,
                        )
                        n_iter = model.n_iter
                        ll_val = np.asarray(
                            model.log_pdf(layer1_data[val_idx], layer2_data[val_idx]),
                            dtype=float,
                        )
                        if not np.all(np.isfinite(ll_val)):
                            raise ValueError("Validation log-likelihood produced non-finite values.")

                        fold_nll = float(-np.sum(ll_val))
                        if score_metric == "nll":
                            fold_score = fold_nll
                            fold_bic = np.nan
                        else:
                            fold_bic = float(model.bic_score(layer1_data[val_idx], layer2_data[val_idx]))
                            if not np.isfinite(fold_bic):
                                raise ValueError("Validation BIC produced non-finite values.")
                            fold_score = fold_bic

                        rec.update({
                            "success": True,
                            "cv_score": fold_score,
                            "fold_nll": fold_nll,
                            "fold_bic": fold_bic,
                            "n_iter": n_iter,
                        })
                        run_info = {
                            "cv_score": fold_score,
                            "fold_nll": fold_nll,
                            "fold_bic": fold_bic,
                            "fold": int(fold_idx),
                            "restart": restart,
                            "seed": seed,
                        }
                        key = (n_layer1, n_layer2)
                        successful_runs_by_config.setdefault(key, []).append(run_info)
                        successful_runs_by_config_and_fold.setdefault((n_layer1, n_layer2, int(fold_idx)), []).append(
                            run_info
                        )
                    except Exception as exc:  # pragma: no cover - kept for robust search.
                        rec.update({
                            "success": False,
                            "cv_score": failed_cv_score,
                            "fold_nll": np.inf,
                            "fold_bic": np.inf if score_metric == "bic" else np.nan,
                            "n_iter": None,
                            "error": repr(exc),
                        })
                        if fail_fast:
                            raise

                    results.append(rec)

    config_results: list[dict[str, Any]] = []
    for n_layer1 in layer1_candidates:
        for n_layer2 in layer2_candidates:
            successful_folds = 0
            failed_folds = 0
            successful_restarts_total = 0
            failed_restarts_total = 0
            for fold_idx in range(n_folds):
                runs = successful_runs_by_config_and_fold.get((n_layer1, n_layer2, fold_idx), [])
                successful_restarts_total += len(runs)
                failed_restarts_total += n_restarts - len(runs)
                if not runs:
                    failed_folds += 1
                    continue
                successful_folds += 1

            if successful_folds == n_folds:
                all_runs = successful_runs_by_config[(n_layer1, n_layer2)]
                scores = np.asarray([float(r["cv_score"]) for r in all_runs], dtype=float)
                score = _selection_score(
                    scores,
                    selection,
                    higher_is_better=metric_higher_is_better,
                )
                summary = {
                    "n_layer1_components": n_layer1,
                    "n_layer2_components": n_layer2,
                    "successful_restarts": int(successful_restarts_total),
                    "failed_restarts": int(failed_restarts_total),
                    "successful_folds": int(successful_folds),
                    "failed_folds": int(failed_folds),
                    "cv_score_mean": float(np.mean(scores)),
                    "cv_score_median": float(np.median(scores)),
                    "cv_score_std": float(np.std(scores)),
                    "cv_score_min": float(np.min(scores)),
                    "cv_score_max": float(np.max(scores)),
                    "selection_score": float(score),
                    "selection": selection,
                    "score_metric": score_metric,
                    "fold_aggregation": "all_folds_restarts",
                    "success": True,
                }
            else:
                summary = {
                    "n_layer1_components": n_layer1,
                    "n_layer2_components": n_layer2,
                    "successful_restarts": int(successful_restarts_total),
                    "failed_restarts": int(failed_restarts_total),
                    "successful_folds": int(successful_folds),
                    "failed_folds": int(failed_folds),
                    "cv_score_mean": np.inf,
                    "cv_score_median": np.inf,
                    "cv_score_std": np.inf,
                    "cv_score_min": np.inf,
                    "cv_score_max": np.inf,
                    "selection_score": failed_selection_score,
                    "selection": selection,
                    "score_metric": score_metric,
                    "fold_aggregation": "all_folds_restarts",
                    "success": False,
                }
            config_results.append(summary)

    successful_configs = [row for row in config_results if row["success"]]
    if not successful_configs:
        raise RuntimeError("All grid-search runs failed. Set fail_fast=True to surface the first error.")

    if selection_higher_is_better:
        best_config_summary = max(successful_configs, key=lambda row: float(row["selection_score"]))
    else:
        best_config_summary = min(successful_configs, key=lambda row: float(row["selection_score"]))
    best_key = (
        int(best_config_summary["n_layer1_components"]),
        int(best_config_summary["n_layer2_components"]),
    )
    best_runs = successful_runs_by_config[best_key]
    best_run = _select_representative_run(
        best_runs,
        selection=selection,
        selection_score=float(best_config_summary["selection_score"]),
        metric_higher_is_better=metric_higher_is_better,
        selection_higher_is_better=selection_higher_is_better,
        metric_key="cv_score",
    )

    best_seed = int(best_run["seed"])
    best_layer1 = int(best_config_summary["n_layer1_components"])
    best_layer2 = int(best_config_summary["n_layer2_components"])
    best_rng = np.random.RandomState(best_seed)
    best_model = builder(best_layer1, best_layer2, init_layer1, init_layer2, best_rng)
    if best_model.n_layer1_components != best_layer1:
        raise ValueError(
            "model_builder returned a model with inconsistent n_layer1_components during refit: "
            f"expected {best_layer1}, got {best_model.n_layer1_components}."
        )
    if not all(mix.n_components == best_layer2 for mix in best_model.layer2_mixtures):
        raise ValueError("model_builder returned a model with inconsistent layer-2 component counts during refit.")

    best_model.fit(
        layer1_data,
        layer2_data,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
        m_step_case=m_step_case,
        c_step_bool=c_step_bool,
    )
    best_n_iter = best_model.n_iter

    best_cv_score = float(best_config_summary["selection_score"])
    best_bic_refit = float(best_model.bic_score(layer1_data, layer2_data))
    best_config = {
        "n_layer1_components": best_layer1,
        "n_layer2_components": best_layer2,
        "fold": int(best_run["fold"]),
        "restart": int(best_run["restart"]),
        "seed": best_seed,
        "n_iter": best_n_iter,
        "selection": selection,
        "score_metric": score_metric,
        "fold_aggregation": "all_folds_restarts",
        "selection_score": float(best_config_summary["selection_score"]),
        "cv_score_mean": float(best_config_summary["cv_score_mean"]),
        "cv_score_median": float(best_config_summary["cv_score_median"]),
        "cv_score_std": float(best_config_summary["cv_score_std"]),
        "cv_score_min": float(best_config_summary["cv_score_min"]),
        "cv_score_max": float(best_config_summary["cv_score_max"]),
    }

    if metric_higher_is_better:
        results_sorted = sorted(results, key=lambda row: (not row["success"], -row["cv_score"]))
    else:
        results_sorted = sorted(results, key=lambda row: (not row["success"], row["cv_score"]))
    if selection_higher_is_better:
        config_results_sorted = sorted(config_results, key=lambda row: (not row["success"], -row["selection_score"]))
    else:
        config_results_sorted = sorted(config_results, key=lambda row: (not row["success"], row["selection_score"]))
    return {
        "best_model": best_model,
        "best_cv_score": best_cv_score,
        "best_bic_refit": best_bic_refit,
        "best_selection_score": float(best_config_summary["selection_score"]),
        "best_config": best_config,
        "selection": selection,
        "score_metric": score_metric,
        "cv": int(n_folds),
        "shuffle": bool(shuffle),
        "results": results_sorted,
        "config_results": config_results_sorted,
    }
