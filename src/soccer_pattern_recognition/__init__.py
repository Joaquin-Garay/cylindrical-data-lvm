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
    "IsolatedTwoLayerMoM",
    "SoccerTwoLayerMoM",
    "consolidate",
    "add_noise",
    "prepare_data",
    "save_spadl_h5",
    "BaseEmission",
    "GaussianEmission",
    "TwoLayerEmission",
    "SoccerEmission",
    "SoccerEmission2",
    "EmissionHMM",
    "GaussianHMM",
    "TwoLayerHMM",
    "Distribution",
    "Categorical",
    "ExponentialFamily",
    "UnivariateGaussian",
    "MultivariateGaussian",
    "VonMises",
    "VonMisesFisher",
    "Cylindrical",
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
        from .hierarchical import TwoLayerMoM

        globals()[name] = TwoLayerMoM
        return TwoLayerMoM

    if name == "IsolatedTwoLayerMoM":
        from .hierarchical import IsolatedTwoLayerMoM

        globals()[name] = IsolatedTwoLayerMoM
        return IsolatedTwoLayerMoM

    if name == "SoccerTwoLayerMoM":
        from .hierarchical import SoccerTwoLayerMoM

        globals()[name] = SoccerTwoLayerMoM
        return SoccerTwoLayerMoM

    if name in {"consolidate", "add_noise", "prepare_data", "save_spadl_h5"}:
        from .utils import add_noise, consolidate, prepare_data, save_spadl_h5

        _symbols = {
            "consolidate": consolidate,
            "add_noise": add_noise,
            "prepare_data": prepare_data,
            "save_spadl_h5": save_spadl_h5,
        }
        globals().update(_symbols)
        return _symbols[name]

    if name in {
        "BaseEmission",
        "GaussianEmission",
        "TwoLayerEmission",
        "SoccerEmission",
        "SoccerEmission2",
        "EmissionHMM",
        "GaussianHMM",
        "TwoLayerHMM",
    }:
        from .hmm import (
            BaseEmission,
            EmissionHMM,
            GaussianEmission,
            GaussianHMM,
            SoccerEmission,
            SoccerEmission2,
            TwoLayerEmission,
            TwoLayerHMM,
        )

        _symbols = {
            "BaseEmission": BaseEmission,
            "GaussianEmission": GaussianEmission,
            "TwoLayerEmission": TwoLayerEmission,
            "SoccerEmission": SoccerEmission,
            "SoccerEmission2": SoccerEmission2,
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
        "VonMisesFisher",
        "Cylindrical",
        "IndGaussVM",
    }:
        from .distributions import (
            Categorical,
            Cylindrical,
            Distribution,
            ExponentialFamily,
            IndGaussVM,
            MultivariateGaussian,
            UnivariateGaussian,
            VonMises,
            VonMisesFisher,
        )

        _symbols = {
            "Distribution": Distribution,
            "Categorical": Categorical,
            "ExponentialFamily": ExponentialFamily,
            "UnivariateGaussian": UnivariateGaussian,
            "MultivariateGaussian": MultivariateGaussian,
            "VonMises": VonMises,
            "VonMisesFisher": VonMisesFisher,
            "Cylindrical": Cylindrical,
            "IndGaussVM": IndGaussVM,
        }
        globals().update(_symbols)
        return _symbols[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
