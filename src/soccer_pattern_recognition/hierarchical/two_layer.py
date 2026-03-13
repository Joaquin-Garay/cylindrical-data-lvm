"""
Two-layer hierarchical mixture-of-mixtures model.
"""

from __future__ import annotations
from typing import Sequence

import numpy as np

from scipy.special import logsumexp

from ..core import _EPS
from ..distributions import MultivariateGaussian, VonMises
from ..mixtures import MixtureModel
from ..metrics.model_selection import _num_free_params_for_component
from ..utils import (
    add_ellips,
    add_arrow,
)

import matplotlib.pyplot as plt
import matplotsoccer as mps

# grab the default color cycle as a list of hex‐colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class TwoLayerMoM:
    """
    Two-layer mixture-of-mixtures model.

    The first layer is an Exponential Family Mixture Model.
    Each first layer component has an associated second layer mixture in
    `layer2_mixtures`, fitted with soft assignments from the first layer.
    """

    def __init__(self,
                 layer1_mixture: MixtureModel,
                 layer2_mixtures: Sequence[MixtureModel]):
        """
        Initialize a two-layer model.

        Parameters
        ----------
        layer1_mixture : MixtureModel
            Mixture over first-layer features.
        layer2_mixtures : Sequence[MixtureModel]
            One second-layer mixture per first-layer component.

        Raises
        ------
        ValueError
            If the number of second-layer mixtures does not match
            `layer1_mixture.n_components`.
        """
        self.layer1_mixture = layer1_mixture
        self.n_layer1_components = layer1_mixture.n_components
        if len(layer2_mixtures) != self.n_layer1_components:
            raise ValueError(
                "Number of layer-2 mixtures must match number of layer-1 components."
            )

        self.layer2_mixtures = layer2_mixtures

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

    def log_pdf(self, layer1_data: np.ndarray, layer2_data: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood per observation under the two-layer model.

        Parameters
        ----------
        layer1_data : np.ndarray, shape (n_obs, d_layer1)
            First-layer observations.
        layer2_data : np.ndarray, shape (n_obs, d_layer2)
            Second-layer observations.

        Returns
        -------
        np.ndarray, shape (n_obs,)
            Log-density for each observation.
        """
        layer1_pdf = self.layer1_mixture.get_posteriors(layer1_data) + _EPS  # (N,K)
        layer1_pdf *= self.layer1_mixture.pdf(layer1_data)[:, None]
        layer2_log_pdf_array = [self.layer2_mixtures[k].log_pdf(layer2_data)[:, None]  # (N,1)
                             for k in range(self.n_layer1_components)]
        layer2_log_pdf = np.concatenate(layer2_log_pdf_array, axis=1)  # (N,K)
        return logsumexp(np.log(layer1_pdf) + layer2_log_pdf, axis=1)  # (N,)

    def pdf(self, layer1_data: np.ndarray, layer2_data: np.ndarray) -> np.ndarray:
        """
        Compute density per observation under the two-layer model.

        Returns
        -------
        np.ndarray, shape (n_obs,)
            Density values for each observation.
        """
        return np.exp(self.log_pdf(layer1_data, layer2_data))

    def n_free_params(self):
        """
        Return the total number of free parameters in the model.
        """
        layer1_n_params = self.n_layer1_components - 1  # prior parameters
        layer1_n_params += _num_free_params_for_component(self.layer1_mixture.components[0]) * self.n_layer1_components

        layer2_n_params = 0
        for k in range(self.n_layer1_components):
            layer2_mixture = self.layer2_mixtures[k]
            layer2_n_params += layer2_mixture.n_components - 1  # prior parameters
            layer2_n_params += _num_free_params_for_component(layer2_mixture.components[0]) * layer2_mixture.n_components

        return layer2_n_params + layer1_n_params

    def bic_score(self, layer1_data, layer2_data) -> float:
        """
        Compute Bayesian Information Criterion (BIC).

        Lower values indicate a better trade-off between fit and complexity.
        """
        layer1_data = np.asarray(layer1_data, dtype=float)
        layer2_data = np.asarray(layer2_data, dtype=float)

        n_obs = layer1_data.shape[0]
        penalty = np.log(n_obs) * self.n_free_params()
        ll = self.log_pdf(layer1_data, layer2_data).sum()
        return penalty - 2 * ll

    def completed_bic_score(self, layer1_data, layer2_data):
        """
        Compute complete-data BIC using hard component assignments.

        Layer-1 and layer-2 latent assignments are approximated via argmax of
        posterior probabilities.
        """
        layer1_data = np.asarray(layer1_data, dtype=float)
        layer2_data = np.asarray(layer2_data, dtype=float)
        n_obs = layer1_data.shape[0]

        penalty = np.log(n_obs) * self.n_free_params()

        # Location mixture posteriors and assignments
        layer1_posteriors = self.layer1_mixture.get_posteriors(layer1_data) + _EPS
        idx_layer1 = np.argmax(layer1_posteriors, axis=1)  # (n_obs,)

        # Precompute log weights for loc mixture
        log_weights_layer1 = np.log(self.layer1_mixture.weights)
        log_prior_layer1 = log_weights_layer1[idx_layer1]  # (n_obs,)

        log_expfam_layer1 = np.empty(n_obs)
        log_prior_layer2 = np.empty(n_obs)
        log_expfam_layer2 = np.empty(n_obs)

        for j, l1_comp in enumerate(self.layer1_mixture.components):
            mask = (idx_layer1 == j)
            if not np.any(mask):
                continue

            # mask is all loc_data assigned to component j
            layer1_block = layer1_data[mask]
            log_expfam_layer1[mask] = l1_comp.log_pdf(layer1_block)

            # directional mixtures
            layer2_mixture = self.layer2_mixtures[j]
            layer2_block = layer2_data[mask]
            layer2_posteriors_block = layer2_mixture.get_posteriors(layer2_block) + _EPS  # (n_j, K_j)
            idx_layer2_block = np.argmax(layer2_posteriors_block, axis=1)  # (n_j,)

            # Precompute log weights for layer2_mixture
            log_weights_layer2 = np.log(layer2_mixture.weights)
            log_prior_layer2[mask] = log_weights_layer2[idx_layer2_block]

            # Compute layer2 log_pdf
            indices = np.where(mask)[0]
            for local_i, global_i in enumerate(indices):
                k = idx_layer2_block[local_i]
                log_expfam_layer2[global_i] = layer2_mixture.components[k].log_pdf(layer2_data[global_i])

        complete_data_likelihood = (log_prior_layer1 + log_expfam_layer1
                          + log_prior_layer2 + log_expfam_layer2).sum()

        return penalty - 2.0 * complete_data_likelihood

    def plot(self,
        *,
        figsize: float = 6,
        arrow_scale: float = 12.0,
        title: str = "",
        show_title: bool = False,
        save: bool = False,
        file_name: str = None,
        show: bool = True):
        """
        Plot first-layer Gaussian components and second-layer Von Mises means.

        This visualization is available only when layer-1 components are
        `MultivariateGaussian` and layer-2 components are `VonMises`.
        Each layer-1 component is drawn as an ellipse, and each associated
        layer-2 component is drawn as an arrow. Arrow length is proportional
        to Von Mises mean resultant length.
        """
        plot_cond = (isinstance(self.layer1_mixture.components[0], MultivariateGaussian) and
            isinstance(self.layer2_mixtures[0].components[0], VonMises))
        if not plot_cond:
            raise ValueError("Plot only available for MultivariateGaussian -> VonMises Mixture-of-mixtures.")

        ax = mps.field(show=False, figsize=figsize)
        cmap = plt.cm.plasma

        for l1_idx, (layer1_component, layer2_mixture) in enumerate(zip(self.layer1_mixture.components,
                                                                         self.layer2_mixtures)):
            prior = self.layer1_mixture.weights[l1_idx]
            col = cmap(-0.8 * prior + 0.9)
            mean, cov = layer1_component.params
            add_ellips(ax, mean, cov, color=col, alpha=0.5)
            x0, y0 = mean

            for layer2_component in layer2_mixture.components:
                angle_mean, _ = layer2_component.params
                r = layer2_component.mean_length
                length = arrow_scale * r
                dx, dy = np.cos(angle_mean), np.sin(angle_mean)
                add_arrow(ax, x0, y0,
                          length * dx, length * dy,
                          linewidth=0.8)

        if show_title:
            plt.title(title)
        if save:
            plt.savefig(f"{file_name}.pdf", bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
