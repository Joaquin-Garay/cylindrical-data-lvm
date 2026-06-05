"""Run extensive grid searches on hierarchical synthetic data and export config results.

Usage example from a Python console:

import os
import sys
import runpy

os.chdir("/Users/jgv/PycharmProjects/soccer-pattern-recognition")
sys.path.insert(0, "src")

sys.argv = [
    "03_hierarchical_mixture_extensive.py",
    "--output-csv", "results/synthetic_data/hierarchical_config_results.csv",
    "--n-samples", "12000",
    "--data-seed", "7",
    "--search-seed", "42",
    "--noise-gauss-factor", "0.15",
    "--noise-vmf-sigma", "0.05",
    "--grid-cyl", "2,3,4,5,6,7,8,9,10",
    "--grid-indcyl", "2,3,4,5,6,7,8,9,10",
    "--grid-l1", "3,4,5,6",
    "--grid-l2", "1,2,3,4",
    "--n-restarts", "30",
    "--selection", "mean_plus_2std",
    "--cv", "5",
    "--score-metric", "bic",
    "--shuffle",
    "--init", "k-means",
    "--tol", "1e-4",
    "--max-iter", "300",
    "--m-step-case", "bregman",
]

runpy.run_path("experiment/synthetic_data/03_hierarchical_mixture_extensive.py", run_name="__main__");
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import soccer_pattern_recognition as spr
from experiment.synthetic_data import experiment_helper as mod


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / max(norm, 1e-12)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1e-12, None)


def _spd_from_factor(factor: np.ndarray, jitter: float = 0.08) -> np.ndarray:
    factor = np.asarray(factor, dtype=float)
    return factor @ factor.T + float(jitter) * np.eye(factor.shape[0], dtype=float)


def _parse_int_grid(text: str) -> tuple[int, ...]:
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    if len(tokens) == 0:
        raise argparse.ArgumentTypeError("Grid must contain at least one integer.")
    try:
        values = tuple(int(t) for t in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Grid values must be integers.") from exc
    if any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("Grid values must be >= 1.")
    return values


def _build_dataset(
    *,
    n_samples: int,
    data_seed: int,
    noise_gauss_factor: float,
    noise_vmf_sigma: float,
) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(data_seed)
    d_gauss = mod.D_GAUSS
    d_vmf = mod.D_VMF

    l1_means = [
        np.array([-1.0, 0.2, 0.1], dtype=float),
        np.array([-0.2, -0.4, 0.0], dtype=float),
        np.array([0.5, 0.1, -0.2], dtype=float),
        np.array([1.0, -0.2, 0.2], dtype=float),
    ]
    l1_factors = [
        np.array([[1.10, 0.50, 0.20], [0.80, 0.60, 0.10], [0.40, 0.20, 0.80]], dtype=float),
        np.array([[1.00, -0.60, 0.20], [-0.70, 1.00, -0.30], [0.20, -0.50, 0.90]], dtype=float),
        np.array([[1.20, 0.70, -0.20], [0.50, 0.90, 0.40], [-0.10, 0.30, 0.80]], dtype=float),
        np.array([[1.10, -0.50, 0.30], [-0.40, 0.80, 0.50], [0.30, 0.40, 0.90]], dtype=float),
    ]
    layer1_components = [
        spr.MultivariateGaussian(d_gauss, mean=mean, covariance=_spd_from_factor(factor))
        for mean, factor in zip(l1_means, l1_factors)
    ]
    layer1_mixture = spr.MixtureModel(
        layer1_components,
        weights=np.array([0.15, 0.35, 0.30, 0.20], dtype=float),
        init="k-means",
        rng=rng,
    )

    layer2_specs = [
        [(_unit([1.00, 0.25, 0.05]), 6.0), (_unit([0.90, 0.40, 0.15]), 4.5), (_unit([0.80, -0.05, 0.60]), 2.2)],
        [(_unit([0.95, 0.15, -0.05]), 5.5), (_unit([0.88, 0.48, 0.10]), 4.0), (_unit([0.78, -0.20, 0.58]), 2.0)],
        [(_unit([0.92, 0.10, 0.10]), 5.0), (_unit([0.86, 0.42, 0.22]), 3.8), (_unit([0.73, -0.10, 0.67]), 1.8)],
        [(_unit([0.89, 0.30, -0.02]), 4.8), (_unit([0.83, 0.50, 0.20]), 3.5), (_unit([0.70, -0.18, 0.69]), 1.7)],
    ]
    layer2_weights = [
        np.array([0.60, 0.30, 0.10], dtype=float),
        np.array([0.55, 0.30, 0.15], dtype=float),
        np.array([0.45, 0.40, 0.15], dtype=float),
        np.array([0.50, 0.25, 0.25], dtype=float),
    ]
    layer2_mixtures: list[spr.MixtureModel] = []
    for specs, weights in zip(layer2_specs, layer2_weights):
        components = [spr.VonMisesFisher(d_vmf, mu=mu, kappa=kappa) for mu, kappa in specs]
        layer2_mixtures.append(
            spr.MixtureModel(
                components=components,
                weights=weights,
                init="k-means",
                rng=rng,
            )
        )

    generator = spr.TwoLayerMoM(
        layer1_mixture=layer1_mixture,
        layer2_mixtures=layer2_mixtures,
    )
    x = generator.sample(n_samples, rng=rng)
    x_gauss = x[:, :d_gauss]
    x_vmf = x[:, d_gauss:]

    sigma_g = noise_gauss_factor * np.maximum(x_gauss.std(axis=0, ddof=1), 1e-8)
    noise_g = rng.normal(0.0, sigma_g, size=(n_samples, d_gauss))
    noise_v = rng.normal(0.0, noise_vmf_sigma, size=(n_samples, d_vmf))

    x_noisy_gauss = x_gauss + noise_g
    x_noisy_vmf = _unit(x_vmf + noise_v)
    x_noisy = np.concatenate((x_noisy_gauss, x_noisy_vmf), axis=1)

    return {
        "x": x,
        "x_gauss": x_gauss,
        "x_vmf": x_vmf,
        "x_noisy": x_noisy,
        "x_noisy_gauss": x_noisy_gauss,
        "x_noisy_vmf": x_noisy_vmf,
    }


def _run_searches(data: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _run_mixture(seed: int, x_key: str, grid: tuple[int, ...], builder):
        return spr.calibrate_mixture_by_cv_grid_search(
            x=data[x_key],
            n_components_grid=grid,
            cv=args.cv,
            n_restarts=args.n_restarts,
            selection=args.selection,
            score_metric=args.score_metric,
            init=args.init,
            model_builder=builder,
            tol=args.tol,
            max_iter=args.max_iter,
            m_step_case=args.m_step_case,
            random_state=seed,
            fail_fast=args.fail_fast,
            verbose=args.verbose,
            shuffle=args.shuffle,
        )

    def _run_mom(seed: int, xg_key: str, xv_key: str, builder):
        return spr.calibrate_mom_by_cv_grid_search(
            layer1_data=data[xg_key],
            layer2_data=data[xv_key],
            n_layer1_grid=args.grid_l1,
            n_layer2_grid=args.grid_l2,
            cv=args.cv,
            n_restarts=args.n_restarts,
            selection=args.selection,
            score_metric=args.score_metric,
            init_layer1=args.init,
            init_layer2=args.init,
            model_builder=builder,
            m_step_case=args.m_step_case,
            tol=args.tol,
            max_iter=args.max_iter,
            random_state=seed,
            fail_fast=args.fail_fast,
            verbose=args.verbose,
            shuffle=args.shuffle,
        )

    specs = [
        {
            "search_name": "cyl_pure",
            "model": "Cylindrical Mixture",
            "case": "Pure signal",
            "kind": "mixture",
            "runner": lambda seed: _run_mixture(seed, "x", args.grid_cyl, mod.cylindrical_mixture_builder_3d),
        },
        {
            "search_name": "cyl_noisy",
            "model": "Cylindrical Mixture",
            "case": "Noisy",
            "kind": "mixture",
            "runner": lambda seed: _run_mixture(seed, "x_noisy", args.grid_cyl, mod.cylindrical_mixture_builder_3d),
        },
        {
            "search_name": "indcyl_pure",
            "model": "Ind. Cylindrical Mixture",
            "case": "Pure signal",
            "kind": "mixture",
            "runner": lambda seed: _run_mixture(seed, "x", args.grid_indcyl, mod.ind_cylindrical_mixture_builder_3d),
        },
        {
            "search_name": "indcyl_noisy",
            "model": "Ind. Cylindrical Mixture",
            "case": "Noisy",
            "kind": "mixture",
            "runner": lambda seed: _run_mixture(
                seed,
                "x_noisy",
                args.grid_indcyl,
                mod.ind_cylindrical_mixture_builder_3d,
            ),
        },
        {
            "search_name": "mom_pure",
            "model": "Two-layer MoM",
            "case": "Pure signal",
            "kind": "mom",
            "runner": lambda seed: _run_mom(seed, "x_gauss", "x_vmf", mod.mom_builder_3d),
        },
        {
            "search_name": "mom_noisy",
            "model": "Two-layer MoM",
            "case": "Noisy",
            "kind": "mom",
            "runner": lambda seed: _run_mom(seed, "x_noisy_gauss", "x_noisy_vmf", mod.mom_builder_3d),
        },
        {
            "search_name": "iso_mom_pure",
            "model": "Isolated Two-layer MoM",
            "case": "Pure signal",
            "kind": "mom",
            "runner": lambda seed: _run_mom(seed, "x_gauss", "x_vmf", mod.mom_iso_builder_3d),
        },
        {
            "search_name": "iso_mom_noisy",
            "model": "Isolated Two-layer MoM",
            "case": "Noisy",
            "kind": "mom",
            "runner": lambda seed: _run_mom(seed, "x_noisy_gauss", "x_noisy_vmf", mod.mom_iso_builder_3d),
        },
    ]

    config_frames: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []

    for spec in specs:
        seed = int(args.search_seed)
        print(f"[RUN] {spec['search_name']} (seed={seed})")
        try:
            search = spec["runner"](seed)
            cfg = pd.DataFrame(search["config_results"]).copy()
            cfg["search_name"] = spec["search_name"]
            cfg["model"] = spec["model"]
            cfg["case"] = spec["case"]
            cfg["selection"] = search["selection"]
            cfg["score_metric"] = search["score_metric"]
            cfg["cv"] = int(search["cv"])
            cfg["best_bic_refit"] = float(search["best_bic_refit"])
            cfg["best_selection_score"] = float(search["best_selection_score"])
            cfg["data_seed"] = int(args.data_seed)
            cfg["search_seed"] = seed
            cfg["n_restarts"] = int(args.n_restarts)
            cfg["max_iter"] = int(args.max_iter)
            cfg["tol"] = float(args.tol)
            cfg["m_step_case"] = args.m_step_case
            cfg["init"] = args.init
            cfg["shuffle"] = bool(args.shuffle)

            best_cfg = search["best_config"]
            if spec["kind"] == "mixture":
                cfg["is_selected_config"] = cfg["n_components"].eq(int(best_cfg["n_components"]))
                cfg["n_layer1_components"] = cfg["n_components"].astype(int)
                cfg["n_layer2_components"] = 0
                cfg = cfg.drop(columns=["n_components"])
            else:
                cfg["is_selected_config"] = (
                    cfg["n_layer1_components"].eq(int(best_cfg["n_layer1_components"]))
                    & cfg["n_layer2_components"].eq(int(best_cfg["n_layer2_components"]))
                )

            config_frames.append(cfg)
            print(
                f"[OK ] {spec['search_name']}: best_bic_refit={search['best_bic_refit']:.3f}, "
                f"selection_score={search['best_selection_score']:.3f}"
            )
        except Exception as exc:  # pragma: no cover
            errors.append(
                {
                    "search_name": spec["search_name"],
                    "model": spec["model"],
                    "case": spec["case"],
                    "search_seed": seed,
                    "error": repr(exc),
                }
            )
            print(f"[ERR] {spec['search_name']}: {exc!r}")
            if args.fail_fast:
                raise

    if not config_frames:
        raise RuntimeError("No successful searches were produced.")

    all_configs = pd.concat(config_frames, ignore_index=True, sort=False)
    all_configs = all_configs.sort_values(
        by=["model", "case", "selection_score"],
        na_position="last",
    ).reset_index(drop=True)

    if "n_components" in all_configs.columns:
        all_configs = all_configs.drop(columns=["n_components"])

    leading_columns = ["case", "model", "n_layer1_components", "n_layer2_components"]
    cv_score_columns = [c for c in all_configs.columns if c.startswith("cv_score")]
    remaining_columns = [
        c for c in all_configs.columns
        if c not in leading_columns and c not in cv_score_columns
    ]
    ordered_columns = [c for c in leading_columns if c in all_configs.columns] + cv_score_columns + remaining_columns
    all_configs = all_configs.loc[:, ordered_columns]

    return all_configs, pd.DataFrame(errors)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample synthetic hierarchical data, run extensive grid searches for multiple "
            "model families, and export all aggregated config_results to CSV."
        )
    )
    parser.add_argument("--output-csv", type=Path, default=Path("results/synthetic_data/hierarchical_config_results.csv"))
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--data-seed", type=int, default=7)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--noise-gauss-factor", type=float, default=0.15)
    parser.add_argument("--noise-vmf-sigma", type=float, default=0.05)

    parser.add_argument("--grid-cyl", type=_parse_int_grid, default=(2, 3, 4, 5, 6, 7, 8))
    parser.add_argument("--grid-indcyl", type=_parse_int_grid, default=(2, 3, 4, 5, 6, 7, 8, 9, 10))
    parser.add_argument("--grid-l1", type=_parse_int_grid, default=(2, 3, 4, 5, 6))
    parser.add_argument("--grid-l2", type=_parse_int_grid, default=(1, 2, 3))

    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument(
        "--selection",
        type=str,
        default="mean",
        choices=("mean", "median", "best", "mean_plus_2std"),
    )
    parser.add_argument("--cv", type=int, default=1)
    parser.add_argument("--score-metric", type=str, default="bic", choices=("nll", "bic"))
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--init", type=str, default="k-means")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--m-step-case", type=str, default="bregman")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    start_time = time.perf_counter()
    parser = _build_parser()
    args: argparse.Namespace | None = None
    try:
        args = parser.parse_args()

        if args.n_samples < 1:
            raise ValueError("--n-samples must be >= 1.")
        if args.n_restarts < 1:
            raise ValueError("--n-restarts must be >= 1.")
        if args.cv < 1:
            raise ValueError("--cv must be >= 1.")

        print("[INFO] Building synthetic dataset...")
        data = _build_dataset(
            n_samples=int(args.n_samples),
            data_seed=int(args.data_seed),
            noise_gauss_factor=float(args.noise_gauss_factor),
            noise_vmf_sigma=float(args.noise_vmf_sigma),
        )
        print("[INFO] Running extensive searches...")
        all_configs, errors = _run_searches(data, args)

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        all_configs.to_csv(args.output_csv, index=False)
        print(f"[DONE] Exported config_results to: {args.output_csv.resolve()}")
        print(f"[INFO] Rows exported: {len(all_configs)}")

        if not errors.empty:
            err_path = args.output_csv.with_name(args.output_csv.stem + "_errors.csv")
            errors.to_csv(err_path, index=False)
            print(f"[WARN] Some searches failed. Exported errors to: {err_path.resolve()}")
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"[INFO] Total execution time: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")


if __name__ == "__main__":
    main()
