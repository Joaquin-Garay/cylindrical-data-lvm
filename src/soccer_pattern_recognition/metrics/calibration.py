"""Calibration helpers for model selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np

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


def calibrate_mom_by_bic_grid_search(
    layer1_data,
    layer2_data,
    *,
    n_layer1_grid: Sequence[int] = (2, 3, 4),
    n_layer2_grid: Sequence[int] = (1, 2),
    n_restarts: int = 2,
    init_layer1: str = "k-means++",
    init_layer2: str = "k-means++",
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
) -> dict[str, Any]:
    """
    Run a small grid-search calibration for a two-layer MoM using BIC as criterion.

    Lower BIC is better. The function fits one model per grid point and restart,
    then returns the best fitted model and metadata.
    """
    layer1_data = np.asarray(layer1_data, dtype=float)
    layer2_data = np.asarray(layer2_data, dtype=float)
    if layer1_data.ndim != 2 or layer2_data.ndim != 2:
        raise ValueError("layer1_data and layer2_data must be 2D arrays.")
    if layer1_data.shape[0] != layer2_data.shape[0]:
        raise ValueError("layer1_data and layer2_data must have the same number of samples.")
    if not isinstance(n_restarts, (int, np.integer)) or int(n_restarts) < 1:
        raise ValueError("n_restarts must be an integer >= 1.")
    n_restarts = int(n_restarts)

    layer1_candidates = _validate_component_grid("n_layer1_grid", n_layer1_grid)
    layer2_candidates = _validate_component_grid("n_layer2_grid", n_layer2_grid)

    builder = _default_two_layer_mom_builder if model_builder is None else model_builder
    if not callable(builder):
        raise TypeError("model_builder must be callable.")

    master_rng = np.random.RandomState(random_state)
    results: list[dict[str, Any]] = []

    best_bic = np.inf
    best_model: Optional["TwoLayerMoM"] = None
    best_config: Optional[dict[str, Any]] = None

    for n_layer1 in layer1_candidates:
        for n_layer2 in layer2_candidates:
            for restart in range(n_restarts):
                seed = int(master_rng.randint(np.iinfo(np.int32).max))
                run_rng = np.random.RandomState(seed)
                rec: dict[str, Any] = {
                    "n_layer1_components": n_layer1,
                    "n_layer2_components": n_layer2,
                    "restart": restart,
                    "seed": seed,
                }

                try:
                    model = builder(n_layer1, n_layer2, init_layer1, init_layer2, run_rng)
                    n_iter = model.fit(
                        layer1_data,
                        layer2_data,
                        tol=tol,
                        max_iter=max_iter,
                        verbose=verbose,
                        m_step_case=m_step_case,
                        c_step_bool=c_step_bool,
                    )
                    bic = float(model.bic_score(layer1_data, layer2_data))
                    rec.update({"success": True, "bic": bic, "n_iter": int(n_iter)})

                    if bic < best_bic:
                        best_bic = bic
                        best_model = model
                        best_config = {
                            "n_layer1_components": n_layer1,
                            "n_layer2_components": n_layer2,
                            "restart": restart,
                            "seed": seed,
                            "n_iter": int(n_iter),
                        }
                except Exception as exc:  # pragma: no cover - kept for robust search.
                    rec.update({"success": False, "bic": np.inf, "n_iter": None, "error": repr(exc)})
                    if fail_fast:
                        raise

                results.append(rec)

    if best_model is None or best_config is None:
        raise RuntimeError("All grid-search runs failed. Set fail_fast=True to surface the first error.")

    results_sorted = sorted(results, key=lambda row: (not row["success"], row["bic"]))
    return {
        "best_model": best_model,
        "best_bic": float(best_bic),
        "best_config": best_config,
        "results": results_sorted,
    }


def calibrate_mixture_by_bic_grid_search(
    x,
    *,
    n_components_grid: Sequence[int] = (2, 3, 4),
    n_restarts: int = 2,
    init: str = "k-means++",
    model_builder: Optional[
        Callable[[int, str, np.random.RandomState], "MixtureModel"]
    ] = None,
    tol: float = 1e-4,
    max_iter: int = 300,
    m_step_case: str = "classic",
    c_step_bool: bool = False,
    verbose: bool = False,
    random_state: Optional[int] = 42,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """
    Run a small grid-search calibration for a MixtureModel using BIC as criterion.

    Lower BIC is better. The function fits one model per grid point and restart,
    then returns the best fitted model and metadata.

    Notes
    -----
    - For non-Gaussian/custom component families, pass ``model_builder``.
    - ``model_builder`` must return a fresh MixtureModel for each call.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim not in {1, 2}:
        raise ValueError("x must be a 1D or 2D array.")
    if x.shape[0] < 1:
        raise ValueError("x must contain at least one sample.")
    if not isinstance(n_restarts, (int, np.integer)) or int(n_restarts) < 1:
        raise ValueError("n_restarts must be an integer >= 1.")
    n_restarts = int(n_restarts)

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
    results: list[dict[str, Any]] = []

    best_bic = np.inf
    best_model: Optional["MixtureModel"] = None
    best_config: Optional[dict[str, Any]] = None

    for n_components in n_candidates:
        for restart in range(n_restarts):
            seed = int(master_rng.randint(np.iinfo(np.int32).max))
            run_rng = np.random.RandomState(seed)
            rec: dict[str, Any] = {
                "n_components": n_components,
                "restart": restart,
                "seed": seed,
            }

            try:
                model = builder(n_components, init, run_rng)
                if model.n_components != n_components:
                    raise ValueError(
                        "model_builder returned a model with inconsistent n_components: "
                        f"expected {n_components}, got {model.n_components}."
                    )

                model.fit(
                    x,
                    tol=tol,
                    max_iter=max_iter,
                    verbose=verbose,
                    m_step_case=m_step_case,
                    c_step_bool=c_step_bool,
                )
                bic = float(model.bic_score(x))
                n_iter = int(model.logger[1]) if isinstance(model.logger, tuple) else None
                rec.update({"success": True, "bic": bic, "n_iter": n_iter})

                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_config = {
                        "n_components": n_components,
                        "restart": restart,
                        "seed": seed,
                        "n_iter": n_iter,
                    }
            except Exception as exc:  # pragma: no cover - kept for robust search.
                rec.update({"success": False, "bic": np.inf, "n_iter": None, "error": repr(exc)})
                if fail_fast:
                    raise

            results.append(rec)

    if best_model is None or best_config is None:
        raise RuntimeError("All grid-search runs failed. Set fail_fast=True to surface the first error.")

    results_sorted = sorted(results, key=lambda row: (not row["success"], row["bic"]))
    return {
        "best_model": best_model,
        "best_bic": float(best_bic),
        "best_config": best_config,
        "results": results_sorted,
    }
