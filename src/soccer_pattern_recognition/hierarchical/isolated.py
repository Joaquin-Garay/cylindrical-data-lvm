"""Isolated two-stage hierarchical mixture-of-mixtures model."""

from __future__ import annotations

import numpy as np

from .two_layer import TwoLayerMoM
from ..core import _EPS


class IsolatedTwoLayerMoM(TwoLayerMoM):
    """Two-layer MoM variant whose ``fit`` uses the isolated two-stage routine."""

    def fit(self,
            layer1_data: np.ndarray,
            layer2_data: np.ndarray,
            tol: float = 1e-4,
            max_iter: int = 1000,
            verbose: bool = False,
            m_step_case: str = "classic",
            c_step_bool: bool = False,
            ) -> int:
        """
        Fit the two-layer model.

        The first-layer mixture is fitted first. Then each second-layer mixture
        is fitted on all second-layer observations, weighted by the posterior
        responsibilities of its corresponding first-layer component.

        Parameters
        ----------
        layer1_data : np.ndarray, shape (n_obs, d_layer1)
            First-layer observations.
        layer2_data : np.ndarray, shape (n_obs, d_layer2)
            Second-layer observations aligned with `layer1_data`.
        tol : float, default=1e-4
            EM convergence tolerance.
        max_iter : int, default=1000
            Maximum EM iterations per mixture.
        verbose : bool, default=False
            If True, print optimization progress.
        m_step_case : str, default="classic"
            M-step variant forwarded to underlying mixtures.
        c_step_bool : bool, default=False
            If True, use classification EM where supported.

        Returns
        -------
        int
            Total number of EM iterations (layer 1 + all layer-2 mixtures).

        Raises
        ------
        ValueError
            If first-layer and second-layer sample counts mismatch, or if
            classification EM is requested with incompatible initialization.
        """

        if c_step_bool and not all(m.init == "k-means" for m in self.layer2_mixtures):
            raise ValueError(
                "Classification EM requires 'k-means' initialization for all layer-2 mixtures."
            )

        layer1_data = np.asarray(layer1_data, dtype=float)
        layer2_data = np.asarray(layer2_data, dtype=float)
        n_obs = layer1_data.shape[0]
        if n_obs != layer2_data.shape[0]:
            raise ValueError("layer1_data and layer2_data must have the same number of samples.")

        _, layer1_counter = self.layer1_mixture.fit(layer1_data,
                                            sample_weight=None,
                                            tol=tol,
                                            max_iter=max_iter,
                                            verbose=verbose,
                                            m_step_case=m_step_case,
                                            c_step_bool=c_step_bool)

        # include a jitter in the posteriors probabilities
        layer1_posteriors = self.layer1_mixture.get_posteriors(layer1_data) + _EPS

        # C-step: One-hot encoding of posterior matrix
        # if c_step:
        #     idx = np.argmax(layer1_posteriors, axis=1)  # shape (N,)
        #     one_hot = np.zeros_like(layer1_posteriors, dtype=float)
        #     one_hot[np.arange(layer1_posteriors.shape[0]), idx] = 1.0
        #     if np.any(one_hot.sum(axis=0) == 0):
        #         # there is an empty cluster
        #         raise ValueError("Empty cluster")
        #     else:
        #         layer1_posteriors = one_hot

        layer2_counter = 0
        for l1_comp in range(self.n_layer1_components):
            _, l2_comp_counter = self.layer2_mixtures[l1_comp].fit(layer2_data,
                                                              sample_weight=layer1_posteriors[:, l1_comp],
                                                              tol=tol,
                                                              max_iter=max_iter,
                                                              verbose=verbose,
                                                              m_step_case=m_step_case,
                                                              c_step_bool=c_step_bool,
                                                              )
            layer2_counter += l2_comp_counter

        return layer1_counter + layer2_counter
