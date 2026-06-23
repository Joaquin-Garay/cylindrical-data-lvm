"""Synthetic data experiment helpers."""

from .alignment import align_cylindrical_mixture_to_generator
from .builders import (
    cylindrical_mixture_builder,
    ind_cylindrical_mixture_builder,
    mom_builder,
    mom_iso_builder,
    train_all_models,
)
from .common import (
    _extract_cross_corr_matrices,
    _extract_cylindrical_gaussian_params,
    _is_mom_like,
    _mom_cross_corr_matrices,
    _safe_weights,
    unit,
)
from .plots import (
    _plot_cross_corr_matrix_grid,
    _set_axes_equal_2d,
    _set_axes_equal_3d,
    _sync_2d_axes,
    plot_cross_corr_matrices,
    plot_cross_corr_model_vs_generator,
    plot_cylindrical_components_2d,
    plot_cylindrical_components_3d,
    plot_mixing_weights_model_vs_generator,
    plot_model_vs_generator_2d,
    plot_mom_components_2d,
    plot_mom_components_3d,
)

__all__ = [
    "align_cylindrical_mixture_to_generator",
    "cylindrical_mixture_builder",
    "ind_cylindrical_mixture_builder",
    "mom_builder",
    "mom_iso_builder",
    "plot_cross_corr_matrices",
    "plot_cross_corr_model_vs_generator",
    "plot_cylindrical_components_2d",
    "plot_cylindrical_components_3d",
    "plot_mixing_weights_model_vs_generator",
    "plot_model_vs_generator_2d",
    "plot_mom_components_2d",
    "plot_mom_components_3d",
    "train_all_models",
    "unit",
    "_extract_cross_corr_matrices",
    "_extract_cylindrical_gaussian_params",
    "_is_mom_like",
    "_mom_cross_corr_matrices",
    "_plot_cross_corr_matrix_grid",
    "_safe_weights",
    "_set_axes_equal_2d",
    "_set_axes_equal_3d",
    "_sync_2d_axes",
]
