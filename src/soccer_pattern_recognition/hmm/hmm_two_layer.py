"""
Two-layer Mixture-of-Mixture Emission and HMM implementations
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state


Array = np.ndarray

from .base_emission import BaseEmission
from .base import EmissionHMM
from ..hierarchical import TwoLayerMoM
from ..mixtures import initialize_model
from ..mixtures import MixtureModel
from ..distributions import MultivariateGaussian, VonMises

ACTION_MAP = {'clearance': 0,
              'corner': 1,
              'cross': 2,
              'dribble': 3,
              'freekick': 4,
              'goalkick': 5,
              'keeper_action': 6,
              'pass': 7,
              'shot': 8,
              'throw_in': 9}

N_POSSIBLE_ACTIONS = len(ACTION_MAP)

class TwoLayerEmission(BaseEmission):
    """
    Toy-model. Soccer pattern recognition two-layer scheme.
    Emission distribution:
        p(x_n|z_n) = Cat(a_n|z_n) * MoM(x-axis_n, y-axis_n, cos_n, sin_n | z_n, a_n),
    with 'Cat' categorical distribution and 'Mom' Mixture-of-mixtures distribution.

    Observable variable x_n is a tuple of (a_n, x-axis_n, y-axis_n, cos_n, sin_n):
        - a_n: action types
        - x-axis_n: x axis action location
        - y-axis_n: y axis action location
        - cos_n: cosine of action direction angle
        - sin_n sine of action direction angle

    Indices and variables:
    - n in {0, ..., N} observations (observable tuple x_n)
    - i in {0, ..., I} HMM states (latent categorical z_n)
    - h in {0, ..., H} action types (observable categorical a_n)
    - j in {0, ..., K_1} 1st-layer mixture components (latent categorical y_n)
    - k in {0, ..., K_2} 2nd-layer mixture components (latent categorical \tilde{y}_n )

    Sufficient statistics:
    - mix_layer1_post: sum_n gamma_n(i) * r1_n,h(i,j)
    - mix_layer1_loc: sum_n gamma_n(i) * r1_n,h(i,j) * vec(x-axis_n, y-axis_n)
    - mix_layer1_loc**2: sum_n gamma_n(i) * r1_n,h(i,j) * vec(x-axis_n, y-axis_n) @ vec(x-axis_n, y-axis_n).T
    - mix_layer2_post: sum_n gamma_n(i) * r1_n,h(i,j) * r2_n,h(i,j,k)
    - mix_layer2_cos: sum_n gamma_n(i) * r1_n,h(i,j) * r2_n,h(i,j,k) * cos_n
    - mix_layer2_sin: sum_n gamma_n(i) * r1_n,h(i,j) * r2_n,h(i,j,k) * sin_n
    """

    param_symbols = "am"

    def __init__(self, emission_hyperparams: dict):
        """
        emission_params: dict
            example {"pass": [3,"k-means", 2, "k-means"],
                    ""dribble": [5,"k-means", 3, "k-means++"]}

            Key: Action types
            Value[0]: int. Number of components in layer 1 (location)
            Value[1]: str. Initialization method in layer 1
            Value[2]: number of components in layer 2 (direction)
            Value[3]: str. Initialization method in layer 2

        Note: All of the 2nd layer cluster will have the same amount of components.

        """
        super().__init__()
        self.hyperparams = emission_hyperparams
        self.n_actions = len(emission_hyperparams)
        self.action_pi_ = {}
        self.action_mom_ = {}

        self.ACTION_MAP = ACTION_MAP
        self.N_POSSIBLE_ACTIONS = N_POSSIBLE_ACTIONS

    @staticmethod
    def get_gaussian_sufficient_stat(x: Array) -> Array:
        """
        Get the sufficient statistic vector e.g. case d=2: [x y x^2 xy yx y^2]
        :return: array of shape (N,d+d^2)
        """
        n = x.shape[0]
        d = x.shape[1]
        outer = np.einsum('ij,ik->ijk', x, x)  # (n,d,d)
        return np.concatenate([x, outer.reshape(n, d ** 2)], axis=1)

    def get_n_fit_scalars_per_param(self) -> Mapping[str, int]:
        """
        Free parameters of the model
        """
        p = 0
        for mom in self.action_mom_.values():
            p += mom.n_free_params()

        return {"a" : self.n_actions - 1,
                "m" : p}

    def initialize(self, x: Array, init_params: str, random_state: Any) -> None:
        """
        I need the full multi-sequence X, otherwise some actions might be empty.
        """
        total_n_obs = x.shape[0]
        for action, value in self.hyperparams.items():
            x_action = x[x[:, 0] == self.ACTION_MAP[action]]
            n_obs = x_action.shape[0]
            if n_obs == 0:
                raise ValueError(f"Not enough data for {action}.")

            self.action_pi_[action] = float(n_obs / total_n_obs)

            self.action_mom_[action] = TwoLayerMoM(
                loc_mixture=MixtureModel(
                            [MultivariateGaussian() for _ in range(value[0])],
                            init=value[1]
                            ),
                dir_mixtures=[
                    MixtureModel( [VonMises() for _ in range(value[2])],
                                init=value[3])
                     for _ in range(value[0])
                    ]
                )
            initialize_model(model=self.action_mom_[action],
                             x=x_action,
                             sample_weight=np.full(n_obs, 1.0 / n_obs, dtype=float))


    def check(self) -> None:
        raise NotImplementedError("TwoLayerEmission is not implemented yet.")

    def compute_log_likelihood(self, x: Array) -> Array:
        """
        Emission data log likelihood given state.
        """
        ll = np.zeros((x.shape[0],))
        for action in self.hyperparams.keys():
            action_idx = self.ACTION_MAP[action]
            indices = np.where(x[:,0] == action_idx)[0]
            x_action = x[indices]
            ll[indices] = (
                    np.log(self.action_pi_[action]) #scalar
                    + self.action_mom_[action].log_pdf(x_action[:, 1:]) #vector
            )

        return ll


    def initialize_sufficient_statistics(self) -> dict[str, Any]:
        n_states, n_features = self._require_binding()
        return {
            "hmm_post_sum": np.zeros((n_states,), dtype=float),

            "action_post": {action: np.zeros((n_states,), dtype=float)
                            for action in self.hyperparams.keys()},

            "mix_layer1_post": {action: np.zeros((n_states, value[0]),
                                                 dtype=float)
                            for action,value in self.hyperparams.items()},

            "mix_layer1_loc": {action: np.zeros((n_states, value[0], 2),
                                                dtype=float)
                            for action,value in self.hyperparams.items()},

            "mix_layer1_loc**2": {action: np.zeros((n_states, value[0], 2, 2),
                                                   dtype=float)
                            for action,value in self.hyperparams.items()},

            "mix_layer2_post": {action: np.zeros((n_states, value[0], value[2]),
                                                 dtype=float)
                            for action,value in self.hyperparams.items()},

            "mix_layer2_dir": {action: np.zeros((n_states, value[0], value[2], 2),
                                                dtype=float)
                            for action,value in self.hyperparams.items()},
        }


    def accumulate_sufficient_statistics(
        self,
        stats: dict[str, Any],
        x: Array,
        posteriors: Array, #gamma hmm posterior
        params: str,
    ) -> None:
        """
        Update sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by self.initialize_sufficient_statistics`.

        x : array, shape (n_samples, n_features)
            Sample sequence.

        lattice : array, shape (n_samples, n_components)
            Probabilities OR Log Probabilities of each sample
            under each of the model states.  Depends on the choice
            of implementation of the Forward-Backward algorithm

        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.

        fwdlattice, bwdlattice : array, shape (n_samples, n_components)
            forward and backward probabilities.

        x: Array shape (n_obs, 5)
        Observable variable x_n is a tuple of (a_n, x-axis_n, y-axis_n, cos_n, sin_n).
        """
        n_states, n_features = self._require_binding()
        posteriors = np.asarray(posteriors, dtype=float)
        sum_hmm_post = posteriors.sum(axis=0)

        if n_features != 5:
            raise ValueError("TwoLayerEmission must have 5 features.")
        if sum_hmm_post <= 0.0:
            raise RuntimeError("Sum of gamma is not positive")

        stats['hmm_post_sum'] += sum_hmm_post

        for action in self.hyperparams.keys():
            action_idx = self.ACTION_MAP[action]
            indices = np.where(x[:, 0] == action_idx)[0]
            x_action = x[indices]
            stats["action_post"][action] += posteriors[indices].sum(axis=0)
            #TODO: The rest...
            # compute responsabilities r1_n,h(i,j)
            self.action_mom_[action].loc_mixture.get_posterior(x_action[:,1:3])
            stats["mix_layer1_post"]
        # compute responsabilities r2_n,h(i,j,k)



    def m_step(self, stats: dict[str, Any], params: str) -> None:
        raise NotImplementedError("TwoLayerEmission is not implemented yet.")

    def sample_from_state(self, state: int, random_state: Any) -> Array:
        raise NotImplementedError("TwoLayerEmission is not implemented yet.")