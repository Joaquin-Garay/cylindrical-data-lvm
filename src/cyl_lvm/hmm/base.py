"""Generic HMM that delegates emission logic to a pluggable emission object."""

from __future__ import annotations

from typing import Any

import numpy as np
from hmmlearn.base import BaseHMM
from sklearn.utils import check_random_state

from ..core.types import Array
from .emissions import BaseEmission


class EmissionHMM(BaseHMM):
    """
    Hidden Markov Model with pluggable emission object.

    The transition/start probabilities are handled by `hmmlearn.BaseHMM`,
    while emission likelihoods and emission parameter updates are delegated to
    `self.emission`.
    """

    def __init__(
        self,
        n_components: int = 1,
        emission: BaseEmission | None = None,

        startprob_prior: Any = 1.0,
        transmat_prior: Any = 1.0,
        algorithm: str = "viterbi",
        random_state: Any = None,
        n_iter: int = 10,
        tol: float = 1e-2,
        verbose: bool = False,
        params: str | None = None,
        init_params: str | None = None,
        implementation: str = "log",
    ) -> None:
        if emission is None:
            raise ValueError("emission must be a BaseEmission instance.")
        if not isinstance(emission, BaseEmission):
            raise TypeError("emission must inherit from BaseEmission.")
        self.emission = emission

        emission_symbols = self._validate_emission_symbols(emission.param_symbols)
        params = f"st{emission_symbols}" if params is None else params
        init_params = f"st{emission_symbols}" if init_params is None else init_params
        self._validate_param_codes("params", params, emission_symbols)
        self._validate_param_codes("init_params", init_params, emission_symbols)

        super().__init__(
            n_components=n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
        )

    @staticmethod
    def _validate_emission_symbols(symbols: str) -> str:
        if not isinstance(symbols, str) or not symbols:
            raise ValueError("Emission param_symbols must be a non-empty string.")
        if any(symbol in {"s", "t"} for symbol in symbols):
            raise ValueError("Emission param_symbols cannot include 's' or 't'.")
        if len(set(symbols)) != len(symbols):
            raise ValueError("Emission param_symbols must not contain duplicates.")
        return symbols

    @staticmethod
    def _validate_param_codes(name: str, values: str, emission_symbols: str) -> None:
        if not isinstance(values, str):
            raise TypeError(f"{name} must be a string of parameter symbols.")
        allowed = set("st") | set(emission_symbols)
        unknown = set(values) - allowed
        if unknown:
            unknown_list = ", ".join(sorted(unknown))
            allowed_list = ", ".join(sorted(allowed))
            raise ValueError(
                f"{name} contains unsupported symbols: {unknown_list}. "
                f"Allowed symbols are: {allowed_list}."
            )

    def _get_n_fit_scalars_per_param(self) -> dict[str, int]:
        if hasattr(self, "n_features"):
            self.emission.bind(self.n_components, self.n_features)

        n_scalars = {
            "s": self.n_components - 1,
            "t": self.n_components * (self.n_components - 1),
        }
        emission_scalars = dict(self.emission.get_n_fit_scalars_per_param())

        if {"s", "t"} & set(emission_scalars):
            raise ValueError("Emission parameter symbols cannot include 's' or 't'.")
        for key, value in emission_scalars.items():
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(
                    f"Invalid scalar count for parameter '{key}': {value!r}."
                )
            n_scalars[key] = int(value)

        for symbol in self.emission.param_symbols:
            n_scalars.setdefault(symbol, 0)
        return n_scalars

    def _can_warm_start(self, X: Array) -> bool:
        if X.ndim != 2:
            return False
        if not hasattr(self, "startprob_") or not hasattr(self, "transmat_"):
            return False
        if hasattr(self, "n_features") and self.n_features != X.shape[1]:
            return False
        try:
            self.emission.bind(self.n_components, X.shape[1])
            self.emission.check()
        except Exception:
            return False
        return True

    def fit(self, X, lengths=None,
            *,
            warm_start: bool = False,
            init_params: str = "") -> "EmissionHMM":
        X_array = np.asarray(X, dtype=float)
        if not warm_start:
            super().fit(X_array, lengths=lengths)
            return self
        if not self._can_warm_start(X_array):
            super().fit(X_array, lengths=lengths)
            return self

        original_init_params = self.init_params
        self.init_params = init_params
        try:
            super().fit(X_array, lengths=lengths)
        finally:
            self.init_params = original_init_params
        return self

    def _init(self, X, lengths=None):
        super()._init(X, lengths)
        self.emission.bind(self.n_components, self.n_features)
        init_emission = bool(set(self.init_params) & set(self.emission.param_symbols))
        if init_emission:
            self.emission.initialize(
                X=np.asarray(X, dtype=float),
                init_params=self.init_params,
                random_state=check_random_state(self.random_state),
            )

    def _check(self):
        super()._check()
        self.emission.bind(self.n_components, self.n_features)
        self.emission.check()

    def _compute_log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")
        log_likelihood = np.asarray(self.emission.compute_log_likelihood(X), dtype=float)
        expected_shape = (X.shape[0], self.n_components)
        if log_likelihood.shape != expected_shape:
            raise ValueError(
                "Emission compute_log_likelihood returned shape "
                f"{log_likelihood.shape}, expected {expected_shape}."
            )
        return log_likelihood

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        emission_stats = self.emission.initialize_sufficient_statistics()
        for key in emission_stats:
            if key in stats:
                raise ValueError(
                    f"Emission stats key collision: {key!r} already exists."
                )
        stats.update(emission_stats)
        return stats

    def _accumulate_sufficient_statistics(
        self,
        stats,
        X,
        lattice,
        posteriors,
        fwdlattice,
        bwdlattice,
    ):
        super()._accumulate_sufficient_statistics(
            stats=stats,
            X=X,
            lattice=lattice,
            posteriors=posteriors,
            fwdlattice=fwdlattice,
            bwdlattice=bwdlattice,
        )
        self.emission.accumulate_sufficient_statistics(
            stats=stats,
            X=np.asarray(X, dtype=float),
            posteriors=np.asarray(posteriors, dtype=float),
            params=self.params,
        )

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        self.emission.m_step(stats, self.params)

    def _generate_sample_from_state(self, state, random_state):
        sample = np.asarray(
            self.emission.sample_from_state(state=state, random_state=random_state),
            dtype=float,
        )
        if sample.ndim == 0:
            sample = sample.reshape(1)
        if sample.ndim != 1:
            raise ValueError(
                "Emission sample_from_state must return a 1D sample vector."
            )
        if sample.shape[0] != self.n_features:
            raise ValueError(
                "Emission sample dimensionality mismatch: got "
                f"{sample.shape[0]}, expected {self.n_features}."
            )
        return sample
