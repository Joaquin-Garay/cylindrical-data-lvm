"""Synthetic data experiment helpers."""

from .alignment import align_cylindrical_mixture_to_generator
from .builders import (
    cylindrical_mixture_builder,
    ind_cylindrical_mixture_builder,
    mom_builder,
    mom_iso_builder,
    train_all_models,
    predict_model,
    ari_model,
    score_model,
)
from .common import (
    _extract_cylindrical_gaussian_params,
    _is_mom_like,
    _safe_weights,
    sample_noisy_train_test,
    unit,
)
from .plots import (
    CYLMIX_COMPARISON_METRICS,
    cylmix_comparison_dataframe,
    _set_axes_equal_2d,
    _set_axes_equal_3d,
    _sync_2d_axes,
    plot_components,
    plot_cross_corr_comparison,
    plot_cross_corr_matrices,
    plot_cylmix_comparison_metrics,
    plot_cylindrical_sample_2d,
    plot_mixing_weights_model_vs_generator,
    plot_model_vs_generator,
)

__all__ = [
    "CYLMIX_COMPARISON_METRICS",
    "align_cylindrical_mixture_to_generator",
    "cylindrical_mixture_builder",
    "cylmix_comparison_dataframe",
    "ind_cylindrical_mixture_builder",
    "mom_builder",
    "mom_iso_builder",
    "predict_model",
    "ari_model",
    "score_model",
    "plot_components",
    "plot_cross_corr_comparison",
    "plot_cross_corr_matrices",
    "plot_cylmix_comparison_metrics",
    "plot_cylindrical_sample_2d",
    "plot_mixing_weights_model_vs_generator",
    "plot_model_vs_generator",
    "sample_noisy_train_test",
    "train_all_models",
    "unit",
    "_extract_cylindrical_gaussian_params",
    "_is_mom_like",
    "_safe_weights",
    "_set_axes_equal_2d",
    "_set_axes_equal_3d",
    "_sync_2d_axes",
]
