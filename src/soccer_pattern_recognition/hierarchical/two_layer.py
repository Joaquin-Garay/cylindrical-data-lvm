"""
Two-layer hierarchical mixture-of-mixtures model.
"""

from __future__ import annotations
from typing import Sequence, Optional
from ..core.types import Array

import numpy as np

from scipy.special import logsumexp

from ..core import _EPS, _TINY
from ..mixtures import MixtureModel, initialize_model
from ..metrics.model_selection import _num_free_params_for_component
from ..utils import (
    validate_sample_weight
)

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

    def init(self, layer1_data, layer2_data):
        """
        Initialization method
        """
        layer1_data = np.asarray(layer1_data, dtype=float)
        layer2_data = np.asarray(layer2_data, dtype=float)
        if layer1_data.shape[0] != layer2_data.shape[0]:
            raise ValueError("layer1_data and layer2_data must have the same number of samples.")

        l1_null_sample_weight = validate_sample_weight(layer1_data,
                                                       sample_weight=None)
        initialize_model(self.layer1_mixture,
                         layer1_data,
                         l1_null_sample_weight)

        r1 = self.layer1_mixture.get_posteriors(layer1_data)
        for i in range(self.n_layer1_components):
            initialize_model(self.layer2_mixtures[i],
                             layer2_data,
                             validate_sample_weight(layer2_data,
                                                    r1[:,i] + _EPS)
                             )


    def l1_responsibilities(self,
            layer1_data: Array,
            layer2_data: Array,):
        """
        Compute first-layer posterior responsibilities.

        For each sample ``n`` and first-layer component ``i``, this returns
        ``p(z1=i | x1_n, x2_n)`` using:

        ``p(z1=i | x1, x2) ∝ p(z1=i) p(x1 | z1=i) p(x2 | z1=i)``.

        Here, ``p(x2 | z1=i)`` is the marginal density of ``layer2_data`` under
        the ``i``-th layer-2 mixture.

        Parameters
        ----------
        layer1_data : Array, shape (n_obs, d_layer1)
            First-layer observations.
        layer2_data : Array, shape (n_obs, d_layer2)
            Second-layer observations aligned with ``layer1_data``.

        Returns
        -------
        Array, shape (n_obs, n_layer1_components)
            Posterior matrix where entry ``[n, i]`` equals
            ``p(z1=i | x1_n, x2_n)``.

        Notes
        -----
        If a sample yields a non-finite normalizer (all joint log-probabilities
        are ``-inf``), its row is replaced by a uniform distribution over
        first-layer components.
        """
        # p(z1=i | x1, x2) ∝ p(z1=i) * p(x1 | z1=i) * p(x2 | z1=i)
        log_prior = np.log(np.maximum(self.layer1_mixture.weights, _TINY))[None, :]  # (1, K)
        log_l1 = np.array(
            [self.layer1_mixture.components[i].log_pdf(layer1_data) for i in range(self.n_layer1_components)],
            dtype=float,
        ).T  # (N, K)
        log_l2 = np.array(
            [self.layer2_mixtures[i].log_pdf(layer2_data) for i in range(self.n_layer1_components)],
            dtype=float,
        ).T  # (N, K)

        log_joint = log_prior + log_l1 + log_l2  # (N, K)
        log_den = logsumexp(log_joint, axis=1, keepdims=True)  # (N, 1)

        posterior = np.exp(log_joint - log_den)

        # Degenerate rows can still happen when all terms are -inf.
        bad_rows = ~np.isfinite(log_den[:, 0])
        if np.any(bad_rows):
            posterior[bad_rows] = 1.0 / self.n_layer1_components

        return posterior

    def l2_responsibilities(self, layer2_data: Array):
        """
        Compute second-layer conditional responsibilities for each layer-1 state.

        For each first-layer component ``i`` and second-layer component ``j``,
        this returns ``p(z2=j | z1=i, x2_n)`` using:

        ``p(z2=j | z1=i, x2) ∝ p(z2=j | z1=i) p(x2 | z1=i, z2=j)``.

        Parameters
        ----------
        layer2_data : Array, shape (n_obs, d_layer2)
            Second-layer observations.

        Returns
        -------
        Array, shape (n_obs, K, K2_max)
            Tensor of conditional posteriors where ``K`` is the number of
            first-layer components and ``K2_max`` is the maximum number of
            second-layer components across layer-2 mixtures.

            ``posterior[n, i, j] = p(z2=j | z1=i, x2_n)`` for ``j < K2_i``.
            Entries for padded indices ``j >= K2_i`` are exactly zero.

        Notes
        -----
        If a row ``(n, i, :)`` has a non-finite normalizer (all logits are
        ``-inf``), valid entries ``:K2_i`` are replaced by a uniform
        distribution and padded entries remain zero.
        """
        N = layer2_data.shape[0]
        K = self.n_layer1_components
        Ki = np.array([self.layer2_mixtures[i].n_components for i in range(K)])
        K2 = Ki.max()

        # p(z2=j | z1=i, x2) ∝ p(z2=j | z1=i) * p(x2 | z1=i, z2=j)
        log_post = np.full((N, K, K2), -np.inf, dtype=float)
        for i in range(K):
            log_w = np.log(np.maximum(self.layer2_mixtures[i].weights, _TINY))[None, :]  # (1, Ki[i])
            log_pdf = np.array(
                [self.layer2_mixtures[i].components[j].log_pdf(layer2_data) for j in range(Ki[i])],
                dtype=float,
            ).T  # (N, Ki[i])
            log_post[:, i, :Ki[i]] = log_w + log_pdf

        log_den = logsumexp(log_post, axis=2, keepdims=True)  # (N, K, 1)
        posterior = np.exp(log_post - log_den)

        # Keep padded entries exactly at 0 and repair degenerate (all -inf) rows.
        posterior[:, :, :] = np.where(np.isfinite(posterior), posterior, 0.0)
        for i in range(K):
            bad_rows = ~np.isfinite(log_den[:, i, 0])
            if np.any(bad_rows):
                posterior[bad_rows, i, :Ki[i]] = 1.0 / Ki[i]
                if Ki[i] < K2:
                    posterior[bad_rows, i, Ki[i]:] = 0.0

        return posterior

    def fit(self,
            layer1_data: Array,
            layer2_data: Array,
            tol: float = 1e-4,
            max_iter: int = 1000,
            verbose: bool = False,
            m_step_case: str = "classic",
            c_step_bool: bool = False,
            ) -> int:
        """
        Fit the two-layer mixture-of-mixtures model with joint EM.

        The E-step computes first-layer responsibilities
        ``p(z1 | x1, x2)`` and second-layer conditional responsibilities
        ``p(z2 | z1, x2)``. The M-step then updates layer-1 and layer-2
        parameters in the same iteration.

        Convergence is monitored with a logger of sample-weighted
        observed-data log-likelihood values.
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

        sample_weight = validate_sample_weight(layer1_data, sample_weight=None)
        if (
            not self.layer1_mixture.is_initialized
            or any(not mixture.is_initialized for mixture in self.layer2_mixtures)
        ):
            self.init(layer1_data, layer2_data)

        logger = []
        n_iter = 0
        for n_iter in range(max_iter):
            # E-step
            layer1_posteriors = self.l1_responsibilities(layer1_data, layer2_data)  # (N, K)
            layer2_posteriors = self.l2_responsibilities(layer2_data)  # (N, K, K2_max)

            if c_step_bool:
                # Hard assignment for layer 1
                idx_l1 = np.argmax(layer1_posteriors, axis=1)  # (N,)
                one_hot_l1 = np.zeros_like(layer1_posteriors, dtype=float)
                one_hot_l1[np.arange(n_obs), idx_l1] = 1.0
                if np.any(one_hot_l1.sum(axis=0) == 0):
                    raise ValueError(
                        "Empty cluster detected during layer-1 C-step. "
                        "Try different initialization or reduce n_components."
                    )
                layer1_posteriors = one_hot_l1

            # Logger: sample-weighted observed-data log-likelihood
            logger.append(float(np.dot(sample_weight, self.log_pdf(layer1_data, layer2_data))))

            # M-step: layer 1
            self.layer1_mixture.weights = np.average(
                layer1_posteriors,
                axis=0,
                weights=sample_weight,
            )
            if np.min(self.layer1_mixture.weights) <= _EPS:
                if verbose:
                    print("lifting layer-1 priors...")
                self.layer1_mixture.weights = (
                    self.layer1_mixture.weights + _EPS
                ) / (1 + self.n_layer1_components * _EPS)

            for i, layer1_comp in enumerate(self.layer1_mixture.components):
                layer1_comp.fit(
                    layer1_data,
                    sample_weight=sample_weight * layer1_posteriors[:, i],
                    case=m_step_case,
                )

            # M-step: layer 2
            for i, layer2_mixture in enumerate(self.layer2_mixtures):
                n_layer2_i = layer2_mixture.n_components
                layer2_posteriors_i = layer2_posteriors[:, i, :n_layer2_i]  # (N, Ki)
                layer1_weight_i = sample_weight * layer1_posteriors[:, i]  # (N,)

                total_weight_i = float(layer1_weight_i.sum())
                if total_weight_i <= 0.0:
                    if verbose:
                        print(
                            f"layer-1 component {i} has zero effective mass during M-step; "
                            "keeping previous layer-2 parameters."
                        )
                    continue

                layer2_mixture.weights = np.average(
                    layer2_posteriors_i,
                    axis=0,
                    weights=layer1_weight_i,
                )
                if np.min(layer2_mixture.weights) <= _EPS:
                    if verbose:
                        print(f"lifting layer-2 priors for layer1 component {i}...")
                    layer2_mixture.weights = (
                        layer2_mixture.weights + _EPS
                    ) / (1 + n_layer2_i * _EPS)

                for j, layer2_comp in enumerate(layer2_mixture.components):
                    layer2_comp.fit(
                        layer2_data,
                        sample_weight=layer1_weight_i * layer2_posteriors_i[:, j],
                        case=m_step_case,
                    )

            # Convergence check
            if n_iter > 10 and abs(logger[-1] - logger[-2]) < tol:
                if verbose:
                    print(
                        f"Converged at iter {n_iter}: "
                        f"LL={logger[-1]:.6f}, Delta LL={logger[-1] - logger[-2]:.2e}"
                    )
                break
        else:
            if verbose:
                print("Reached max_iter without full convergence.")

        self.logger_ = logger
        return n_iter

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> Array:
        pass

    def log_pdf(self, layer1_data: Array, layer2_data: Array) -> Array:
        """
        Compute log-likelihood per observation under the two-layer model.

        Parameters
        ----------
        layer1_data : Array, shape (n_obs, d_layer1)
            First-layer observations.
        layer2_data : Array, shape (n_obs, d_layer2)
            Second-layer observations.

        Returns
        -------
        Array, shape (n_obs,)
            Log-density for each observation.
        """
        layer1_pdf = self.layer1_mixture.get_posteriors(layer1_data) + _EPS  # (N,K)
        layer1_pdf *= self.layer1_mixture.pdf(layer1_data)[:, None]
        layer2_log_pdf_array = [self.layer2_mixtures[k].log_pdf(layer2_data)[:, None]  # (N,1)
                             for k in range(self.n_layer1_components)]
        layer2_log_pdf = np.concatenate(layer2_log_pdf_array, axis=1)  # (N,K)
        return logsumexp(np.log(layer1_pdf) + layer2_log_pdf, axis=1)  # (N,)

    def pdf(self, layer1_data: Array, layer2_data: Array) -> Array:
        """
        Compute density per observation under the two-layer model.

        Returns
        -------
        Array, shape (n_obs,)
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

            # mask is all layer1_data assigned to component j
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
