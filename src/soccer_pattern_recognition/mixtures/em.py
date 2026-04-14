""" Expectation-Maximization algorithm for Mixture models"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Sequence

import numpy as np
from scipy.special import logsumexp

from ..core import _EPS
from ..core.types import Array
from .initialization import initialize_model
from ..utils.checks import validate_sample_weight
if TYPE_CHECKING:
    from .mixture import MixtureModel

def e_step(model: "MixtureModel",
            x: Array,
            ) -> Tuple[Array, float, float, Array]:
    """
    Run the EM E-step for an initialized mixture model.

    Computes, in a numerically stable log-space form:
    - posterior responsibilities r_ij = p(z_i=j | x_i)
    - observed-data log-likelihood: sum_i log p(x_i)
    - expected complete-data log-likelihood (Q term):
      sum_i sum_j r_ij [log pi_j + log p(x_i | theta_j)]

    Parameters
    ----------
    model : MixtureModel
        Mixture model with initialized component parameters and weights.
    x : np.ndarray of shape (n_samples,) or (n_samples, n_features)
        Input samples.

    Returns
    -------
    r : np.ndarray of shape (n_samples, n_components)
        Responsibilities for each sample/component.
    log_likelihood : float
        Total observed-data log-likelihood over all samples.
    expected_log_likelihood : float
        Expected complete-data log-likelihood under r.
    log_px : np.ndarray of shape (n_samples,)
        Per-sample log p(x_i).

    Raises
    ------
    RuntimeError
        If model weights are not initialized.
    """
    x = np.asarray(x, dtype=float)
    # take log to prevent underflow
    eps = np.finfo(float).tiny
    log_prior = np.log(model.weights + eps)  # (K,) := log prior_j
    log_p = model.log_pdf_components(x)  # (N, K) := log exp_family(x_i|theta_j)
    log_numerator = log_prior + log_p  # (N, K) := log prior_j + log exp_family(x_i|theta_j)
    log_denominator = logsumexp(log_numerator, axis=1,
                                keepdims=True)  # (N, 1) := log (sum_k (prior_k * exp_family(x_i|theta_k)) ) = log p(x_i)
    log_r = log_numerator - log_denominator  # (N, K) := log(r_{ij})
    r = np.exp(log_r)  # responsibilities (N, K)

    log_likelihood = float(
        log_denominator.sum())  # data log likelihood as sum_i log(sum_j prior_j exp_family(x_i|theta_j))

    expected_log_likelihood = float(np.sum(log_numerator * r))  # expected complete-data log-likelihood

    return r, log_likelihood, expected_log_likelihood, log_denominator.flatten()

def c_step(r: Array):
    """
    Compute hard one-hot responsibilities from a soft responsibility matrix.

    Args:
        r: Soft responsibilities of shape (n_samples, n_components).

    Returns:
        Hard one-hot responsibilities with the same shape as r.

    Raises:
        ValueError: If hard assignment creates an empty component.
    """
    idx = np.argmax(r, axis=1)  # shape (N,)
    one_hot = np.zeros_like(r, dtype=float)
    one_hot[np.arange(r.shape[0]), idx] = 1.0
    if np.any(one_hot.sum(axis=0) == 0):
        raise ValueError(
            "Empty cluster detected during C-step. "
            "Try different initialization or reduce n_components."
        )
    return one_hot

def m_step(model: "MixtureModel",
           r: Array,
           x: Array,
           sample_weight: Sequence[float] = None,
           m_step_case: str = "classic",
           verbose: bool = False) -> None:
    """
    Update mixture weights and component parameters for one M-step.

    Args:
        model: Initialized mixture model updated in place.
        r: Responsibilities of shape (n_samples, n_components).
        x: Input samples of shape (n_samples,) or (n_samples, n_features).
        sample_weight: Optional sample weights of shape (n_samples,).
        m_step_case: Component fitting mode passed to comp.fit.
        verbose: If True, print numerical stabilization messages.

    Returns:
        None.
    """
    x = np.asarray(x, dtype=float)
    sample_weight = validate_sample_weight(x, sample_weight)

    # M-step: Maximize sample-weighted data log likelihood
    # update priors
    model.weights = np.average(r, axis=0, weights=sample_weight)  # (K,)
    # lift the priors when one of them is below 1 basis points
    if np.min(model.weights) <= _EPS:
        if verbose:
            print(f"lifting priors...")
        model.weights = (model.weights + _EPS) / (1 + model.n_components * _EPS)
    # update distribution parameters
    for j, comp in enumerate(model.components):
        comp.fit(x, sample_weight=sample_weight * r[:, j], case=m_step_case)

def fit_em(model: "MixtureModel",
           x: Array,
           sample_weight: Sequence[float] = None,
           tol: float = 1e-4,
           max_iter: int = 1000,
           m_step_case: str = "classic",
           c_step_bool: bool = False,
           verbose: bool = False) -> Tuple[Sequence[float], int]:
    """
    Perform the Expectation-Maximization algorithm to fit a mixture model.
    The model is initialized if needed, then iterated until convergence or max_iter.
    """
    x = np.asarray(x, dtype=float)
    sample_weight = validate_sample_weight(x, sample_weight)
    if not model.is_initialized:
        initialize_model(model, x, sample_weight)

    logger = []
    it = 0
    for it in range(max_iter):
        # E-step: Compute the posterior
        r, _, _, log_likelihood_arr = e_step(model, x)
        # sample-weighted data log likelihood
        logger.append(float(np.dot(sample_weight, log_likelihood_arr)))

        # C-step: One-hot encoding of posterior matrix
        if c_step_bool:
            r = c_step(r)

        # M-step: Maximize sample-weighted data log likelihood
        m_step(model, r, x, sample_weight, m_step_case, verbose)

        # check convergence
        if it > 10 and abs(logger[-1] - logger[-2]) < tol:
            if verbose:
                print(f"Converged at iter {it}: LL={logger[-1]:.2f}, Delta LL={logger[-1] - logger[-2]:.2e}")
            break
    else:
        if verbose:
            print("Reached max_iter without full convergence.")

    return logger, it
