"""Top-level package API for soccer_pattern_recognition."""

from importlib import import_module
from typing import Any

__all__ = [
    "core",
    "distributions",
    "hierarchical",
    "hmm",
    "metrics",
    "mixtures",
    "utils",
    "MixtureModel",
    "TwoLayerMoM",
    "BaseEmission",
    "GaussianEmission",
    "GaussianMixtureEmission",
    "TwoLayerEmission",
    "EmissionHMM",
    "GaussianHMM",
    "TwoLayerHMM",
    "Distribution",
    "Categorical",
    "ExponentialFamily",
    "UnivariateGaussian",
    "MultivariateGaussian",
    "VonMises",
    "IndGaussVM",
]


def __getattr__(name: str) -> Any:
    if name in {
        "core",
        "distributions",
        "hierarchical",
        "hmm",
        "metrics",
        "mixtures",
        "utils",
    }:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    if name == "MixtureModel":
        from .mixtures import MixtureModel

        globals()[name] = MixtureModel
        return MixtureModel

    if name == "TwoLayerMoM":
        from .hierarchical.two_layer import TwoLayerMoM

        globals()[name] = TwoLayerMoM
        return TwoLayerMoM

    if name in {
        "BaseEmission",
        "GaussianEmission",
        "GaussianMixtureEmission",
        "TwoLayerEmission",
        "EmissionHMM",
        "GaussianHMM",
        "TwoLayerHMM",
    }:
        from .hmm import (
            BaseEmission,
            EmissionHMM,
            GaussianEmission,
            GaussianHMM,
            GaussianMixtureEmission,
            TwoLayerEmission,
            TwoLayerHMM,
        )

        _symbols = {
            "BaseEmission": BaseEmission,
            "GaussianEmission": GaussianEmission,
            "GaussianMixtureEmission": GaussianMixtureEmission,
            "TwoLayerEmission": TwoLayerEmission,
            "EmissionHMM": EmissionHMM,
            "GaussianHMM": GaussianHMM,
            "TwoLayerHMM": TwoLayerHMM,
        }
        globals().update(_symbols)
        return _symbols[name]

    if name in {
        "Distribution",
        "Categorical",
        "ExponentialFamily",
        "UnivariateGaussian",
        "MultivariateGaussian",
        "VonMises",
        "IndGaussVM",
    }:
        from .distributions import (
            Categorical,
            Distribution,
            ExponentialFamily,
            IndGaussVM,
            MultivariateGaussian,
            UnivariateGaussian,
            VonMises,
        )

        _symbols = {
            "Distribution": Distribution,
            "Categorical": Categorical,
            "ExponentialFamily": ExponentialFamily,
            "UnivariateGaussian": UnivariateGaussian,
            "MultivariateGaussian": MultivariateGaussian,
            "VonMises": VonMises,
            "IndGaussVM": IndGaussVM,
        }
        globals().update(_symbols)
        return _symbols[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
