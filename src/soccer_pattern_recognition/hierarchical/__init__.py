"""Public API for hierarchical models."""

from __future__ import annotations

from typing import Any

__all__ = ["TwoLayerMoM", "IsolatedTwoLayerMoM", "SoccerTwoLayerMoM"]


def __getattr__(name: str) -> Any:
    if name == "TwoLayerMoM":
        from .two_layer import TwoLayerMoM

        globals()[name] = TwoLayerMoM
        return TwoLayerMoM

    if name == "IsolatedTwoLayerMoM":
        from .isolated import IsolatedTwoLayerMoM

        globals()[name] = IsolatedTwoLayerMoM
        return IsolatedTwoLayerMoM

    if name == "SoccerTwoLayerMoM":
        from .soccer_two_layer import SoccerTwoLayerMoM

        globals()[name] = SoccerTwoLayerMoM
        return SoccerTwoLayerMoM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
