"""Public API for utility helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "add_arrow",
    "add_ellips",
    "add_noise",
    "consolidate",
    "prepare_data",
    "save_spadl_h5",
    "validate_sample_weight",
]


def __getattr__(name: str) -> Any:
    if name == "validate_sample_weight":
        from .checks import validate_sample_weight

        globals()[name] = validate_sample_weight
        return validate_sample_weight

    if name in {"add_arrow", "add_ellips"}:
        from .visualization import add_arrow, add_ellips

        globals()["add_arrow"] = add_arrow
        globals()["add_ellips"] = add_ellips
        return globals()[name]

    if name in {"consolidate", "add_noise", "prepare_data"}:
        from .features import add_noise, consolidate, prepare_data

        globals()["consolidate"] = consolidate
        globals()["add_noise"] = add_noise
        globals()["prepare_data"] = prepare_data
        return globals()[name]

    if name == "save_spadl_h5":
        from .io import save_spadl_h5

        globals()[name] = save_spadl_h5
        return save_spadl_h5

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
