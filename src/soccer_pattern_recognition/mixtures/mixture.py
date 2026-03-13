# mixture.py

"""
Mixture model.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Sequence, Tuple, TypeAlias

from ..distributions import Distribution
from ..distributions.expfam import ExponentialFamily
from .em import e_step, c_step, m_step, fit_em
from .initialization import initialize_model
from scipy.special import logsumexp

Array: TypeAlias = np.ndarray


# -------------------- Mixture Model --------------------
class MixtureModel(Distribution):
    def __init__(self, components: list[ExponentialFamily],
                 weights: Optional[Array] = None,
                 init: Optional[str] = None,
                 rng: Optional[int] = None,
                 ):

        self._components = components
        self._rng = np.random.RandomState(42) if rng is None else np.random.RandomState(rng)

        allowed = {"k-means++", "k-means", "random_from_data", "random"}
        if init is None:
            self._init = "k-means++"
        elif init in allowed:
            self._init = init
        else:
            raise ValueError(f"init must be one of {sorted(allowed)}")

        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.ndim != 1 or weights.size != self.n_components:
                raise ValueError("Components and weights mismatch.")
            if np.any(weights <= 0):
                raise ValueError("All weights must be > 0.")
            self._weights = weights / weights.sum()
            self._is_initialized = True
        else:
            self._weights = None
            self._is_initialized = False

    def _initialize(self,
                    x: Array,
                    sample_weight: Array,
                    ) -> None:
        initialize_model(self, x, sample_weight)

    # ---- Getter and Setters ----
    @property
    def weights(self):
        if self._weights is None:
            raise RuntimeError("Mixture weights are not initialized yet.")
        return self._weights.copy()

    @weights.setter
    def weights(self, weights: Array):
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            raise ValueError("weights must be a 1D array.")
        if w.size != self.n_components:
            raise ValueError(
                f"weights must have length {self.n_components}, got {w.size}."
            )
        if not np.all(np.isfinite(w)):
            raise ValueError("weights contains non-finite values.")
        if np.any(w <= 0.0):
            raise ValueError("All weights must be > 0.")

        total = float(w.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("weights must sum to a finite positive value.")

        self._weights = w / total
        self._is_initialized = True

    @property
    def components(self):
        return self._components

    @property
    def n_components(self) -> int:
        return len(self._components)

    @property
    def init(self):
        return self._init

    @property
    def rng(self):
        return self._rng

    @property
    def is_initialized(self):
        return self._is_initialized

    def get_posteriors(self, x: Array):
        x = np.asarray(x, dtype=float)
        post, _, _, _ = self._e_step(x)
        return post

    def get_data_ll(self, x: Array):
        """
        Data Log-likelihood
        """
        x = np.asarray(x, dtype=float)
        _, log_likelihood, _, _ = self._e_step(x)
        return log_likelihood

    def get_expected_ll(self, x: Array):
        """
        Expected Complete-Data Log-likelihood (EM Q function)
        """
        x = np.asarray(x, dtype=float)
        _, _, expected_log_likelihood, _ = self._e_step(x)
        return expected_log_likelihood

    # ---- Densities ----
    def log_pdf_components(self, x: Array) -> Array:
        """
        Returns log p(x_i | k) for all i,k
        Shape: (N, K)
        """
        x = np.asarray(x, dtype=float)
        return np.column_stack([c.log_pdf(x) for c in self._components])

    def log_pdf(self, x: Array) -> Array:
        """
        Return log p(x)
        Shape: (N,)
        """
        x = np.asarray(x, dtype=float)
        log_pi = np.log(self._weights)  # (K,)
        return logsumexp(self.log_pdf_components(x) + log_pi, axis=1)

    def pdf(self, x):
        return np.exp(self.log_pdf(x))

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        raise NotImplementedError("MixtureModel.sample is not implemented yet.")

    # ---- Expectation Maximization Algorithm ----

    def _e_step(self, x: Array):
        """
        See mixtures/em.py
        """
        return e_step(self, x)

    @staticmethod
    def _c_step(r: Array):
        """
        See mixtures/em.py
        """
        return c_step(r)

    def _m_step(self,
        r: Array,
        x: Array,
        sample_weight: Array = None,
        m_step_case: str = "classic",
        verbose: bool = False):
        """
        See mixtures/em.py
        """
        m_step(self, r, x, sample_weight, m_step_case, verbose)


    def fit(self,
            x: Array,
            sample_weight: Sequence[float] = None,
            tol: float = 1e-4,
            max_iter: int = 1000,
            verbose: bool = False,
            m_step_case: str = "classic",
            c_step_bool: bool = False,
            ) -> Tuple[Sequence[float], int]:
        """
        Perform the Expectation-Maximization algorithm to fit a mixture model.
        It stops as soon as the absolute difference between two iterations is below the tolerance.
        """
        x = np.asarray(x, dtype=float)

        return fit_em(self, x, sample_weight,
                      tol,
                      max_iter,
                      m_step_case,
                      c_step_bool,
                      verbose)


    # ---- Display ----
    @staticmethod
    def _format_component(idx: int, w: float | None, comp) -> str:
        w_str = f"{w:0.3f}" if w is not None else "—"
        return f"  ├─ ({idx}) w={w_str}  {comp!r}"

    def __repr__(self) -> str:
        header = f"{self.__class__.__name__}(n_components={self.n_components})"
        if self._components is None:
            return header + "  [no components]"

        lines = [
            self._format_component(j,
                                   None if self._weights is None else self._weights[j],
                                   comp)
            for j, comp in enumerate(self._components)
        ]
        # Use a unicode corner for the last line
        if lines:
            lines[-1] = lines[-1].replace("├─", "└─", 1)
        return "\n".join([header, *lines])

    def predict_proba(self, x: Array) -> Array:
        """Alias for get_posteriors(x)."""
        return self.get_posteriors(x)

    def predict(self, x: Array) -> Array:
        """Hard labels via argmax of posterior responsibilities."""
        return np.argmax(self.get_posteriors(x), axis=1)

    def score(self, x: Array) -> float:
        """Average log-likelihood per sample (sklearn-style)."""
        x = np.asarray(x, dtype=float)
        return float(self.log_pdf(x).mean())
