"""
Two-layer Mixture-of-Mixture Emission and HMM implementations
"""

from __future__ import annotations

import warnings
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

# ACTION_MAP = {'clearance': 0,
#               'corner': 1,
#               'cross': 2,
#               'dribble': 3,
#               'freekick': 4,
#               'goalkick': 5,
#               'keeper_action': 6,
#               'pass': 7,
#               'shot': 8,
#               'throw_in': 9}

ACTION_MAP = {'starting': 10,
              'pass':7,
              'dribble':3,
              'shot':8}

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

    def __init__(self, emission_hyperparams: dict, ignore_actions: bool = False):
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
        self.ignore_actions = bool(ignore_actions)
        if self.ignore_actions and len(self.hyperparams) != 1:
            raise ValueError(
                "ignore_actions=True requires exactly one action entry in emission_hyperparams."
            )
        self.model_actions = (
            tuple(self.hyperparams.keys())
            if not self.ignore_actions
            else (next(iter(self.hyperparams.keys())),)
        )
        self.n_actions = len(self.model_actions)
        # for each hmm state, store a dictionary of actions: params.
        self.action_pi_ = []
        self.action_mom_ = []

        self.ACTION_MAP = ACTION_MAP
        self.N_POSSIBLE_ACTIONS = N_POSSIBLE_ACTIONS

    def _indices_for_action(self, X: Array, action: str) -> Array:
        if self.ignore_actions:
            return np.arange(X.shape[0], dtype=int)
        action_idx = self.ACTION_MAP[action]
        return np.where(X[:, 0] == action_idx)[0]

    @staticmethod
    def get_gaussian_sufficient_stat(X: Array) -> Array:
        """
        Get the sufficient statistic vector e.g. case d=2: [x y x^2 xy yx y^2]
        :return: array of shape (N,d+d^2)
        """
        n = X.shape[0]
        d = X.shape[1]
        outer = np.einsum('ij,ik->ijk', X, X)  # (n,d,d)
        return np.concatenate([X, outer.reshape(n, d ** 2)], axis=1)

    def get_n_fit_scalars_per_param(self) -> Mapping[str, int]:
        """
        Free parameters of the model
        """
        p = 0
        for mom_dict in self.action_mom_:
            for mom in mom_dict.values():
                p += mom.n_free_params()

        return {"a" : self.n_actions - 1,
                "m" : p}

    def initialize(self, X: Array,
                   init_params: str,
                   random_state: Any) -> None:
        """
        I need the full multi-sequence X, otherwise some actions might be empty.
        """
        n_states, n_features = self._require_binding()
        total_n_obs = X.shape[0]

        if self.action_pi_:
            self.action_pi_.clear()
        if self.action_mom_:
            self.action_mom_.clear()

        for state in range(n_states):
            action_pi_dict = {}
            action_mom_dict = {}
            for action in self.model_actions:
                value = self.hyperparams[action]
                n_comp_layer1, init_layer1, n_comp_layer2, init_layer2 = value
                indices = self._indices_for_action(X, action)
                X_action = X[indices]
                n_obs_in_action = X_action.shape[0]
                if n_obs_in_action == 0:
                    raise ValueError(f"Not enough data for {action}.")

                action_pi_dict[action] = float(n_obs_in_action / total_n_obs)

                layer1_mixture = MixtureModel(
                                [MultivariateGaussian() for _ in range(n_comp_layer1)],
                                init=value[1],
                                rng=random_state,
                                )
                initialize_model(
                    layer1_mixture,
                    X_action[:, 1:3],
                    np.full(n_obs_in_action, 1.0 / n_obs_in_action, dtype=float),
                )

                layer2_mixture_list = []
                for l1_comp in range(n_comp_layer1):
                    layer2_mixture = MixtureModel([VonMises() for _ in range(n_comp_layer2)],
                                init=init_layer2,
                                rng=random_state,
                                )
                    initialize_model(
                        layer2_mixture,
                        X_action[:, 3:5],
                        np.full(n_obs_in_action, 1.0 / n_obs_in_action, dtype=float),
                    )
                    layer2_mixture_list.append(layer2_mixture)

                action_mom_dict[action] = TwoLayerMoM(layer1_mixture=layer1_mixture,
                                                      layer2_mixtures=layer2_mixture_list)


            #store the initialized params
            self.action_pi_.append(action_pi_dict)
            self.action_mom_.append(action_mom_dict)

    def export_params(self) -> dict[str, Any]:
        n_states, _ = self._require_binding()
        self.check()
        payload: dict[str, Any] = {
            "ignore_actions": bool(self.ignore_actions),
            "model_actions": list(self.model_actions),
            "hyperparams": {
                action: [
                    int(value[0]),
                    str(value[1]),
                    int(value[2]),
                    str(value[3]),
                ]
                for action, value in self.hyperparams.items()
            },
            "action_pi": [],
            "action_mom": [],
        }
        for state in range(n_states):
            state_pi: dict[str, float] = {}
            state_mom: dict[str, Any] = {}
            for action in self.model_actions:
                state_pi[action] = float(self.action_pi_[state][action])
                mom = self.action_mom_[state][action]
                layer1 = mom.layer1_mixture
                layer1_components = []
                for component in layer1.components:
                    mean, covariance = component.params
                    layer1_components.append(
                        {
                            "mean": np.asarray(mean, dtype=float).tolist(),
                            "covariance": np.asarray(covariance, dtype=float).tolist(),
                        }
                    )
                layer1_payload = {
                    "weights": np.asarray(layer1.weights, dtype=float).tolist(),
                    "components": layer1_components,
                }
                layer2_payload = []
                for layer2_mix in mom.layer2_mixtures:
                    vm_components = []
                    for vm in layer2_mix.components:
                        loc, kappa = vm.params
                        vm_components.append(
                            {"loc": float(loc), "kappa": float(kappa)}
                        )
                    layer2_payload.append(
                        {
                            "weights": np.asarray(layer2_mix.weights, dtype=float).tolist(),
                            "components": vm_components,
                        }
                    )
                state_mom[action] = {
                    "layer1": layer1_payload,
                    "layer2": layer2_payload,
                }
            payload["action_pi"].append(state_pi)
            payload["action_mom"].append(state_mom)
        return payload

    def set_emission_params(self, params: Mapping[str, Any]) -> None:
        if not isinstance(params, Mapping):
            raise TypeError("params must be a mapping produced by export_params().")
        required_keys = {"hyperparams", "action_pi", "action_mom"}
        missing = sorted(required_keys - set(params.keys()))
        if missing:
            raise ValueError(f"Missing keys in emission params: {missing}.")

        hyperparams_raw = params["hyperparams"]
        if not isinstance(hyperparams_raw, Mapping) or len(hyperparams_raw) == 0:
            raise ValueError("params['hyperparams'] must be a non-empty mapping.")
        hyperparams: dict[str, list[Any]] = {}
        for action, value in hyperparams_raw.items():
            if not isinstance(value, (list, tuple)) or len(value) != 4:
                raise ValueError(
                    f"hyperparams['{action}'] must be [n_comp_layer1, init1, n_comp_layer2, init2]."
                )
            hyperparams[str(action)] = [
                int(value[0]),
                str(value[1]),
                int(value[2]),
                str(value[3]),
            ]

        ignore_actions = bool(params.get("ignore_actions", self.ignore_actions))
        self.ignore_actions = ignore_actions
        self.hyperparams = hyperparams
        if self.ignore_actions and len(self.hyperparams) != 1:
            raise ValueError(
                "ignore_actions=True requires exactly one action entry in emission_hyperparams."
            )
        self.model_actions = (
            tuple(self.hyperparams.keys())
            if not self.ignore_actions
            else (next(iter(self.hyperparams.keys())),)
        )
        self.n_actions = len(self.model_actions)

        if "model_actions" in params:
            exported_model_actions = tuple(params["model_actions"])
            if exported_model_actions != self.model_actions:
                raise ValueError(
                    "params['model_actions'] does not match hyperparams/ignore_actions."
                )

        action_pi_raw = params["action_pi"]
        action_mom_raw = params["action_mom"]
        if not isinstance(action_pi_raw, Sequence) or not isinstance(action_mom_raw, Sequence):
            raise TypeError("params['action_pi'] and params['action_mom'] must be sequences.")
        if len(action_pi_raw) != len(action_mom_raw):
            raise ValueError("params['action_pi'] and params['action_mom'] length mismatch.")

        n_states_from_params = len(action_pi_raw)
        try:
            n_states_bound, _ = self._require_binding()
            if n_states_bound != n_states_from_params:
                raise ValueError(
                    f"State count mismatch: bound model has {n_states_bound}, params contain {n_states_from_params}."
                )
        except RuntimeError:
            pass

        expected_actions = set(self.model_actions)
        action_pi_list: list[dict[str, float]] = []
        action_mom_list: list[dict[str, TwoLayerMoM]] = []

        for state_idx in range(n_states_from_params):
            state_pi_raw = action_pi_raw[state_idx]
            state_mom_raw = action_mom_raw[state_idx]
            if not isinstance(state_pi_raw, Mapping) or not isinstance(state_mom_raw, Mapping):
                raise TypeError(
                    f"State {state_idx}: action_pi/action_mom entries must be mappings."
                )
            if set(state_pi_raw.keys()) != expected_actions:
                raise ValueError(
                    f"State {state_idx}: action_pi keys mismatch expected actions."
                )
            if set(state_mom_raw.keys()) != expected_actions:
                raise ValueError(
                    f"State {state_idx}: action_mom keys mismatch expected actions."
                )

            state_pi: dict[str, float] = {}
            state_mom: dict[str, TwoLayerMoM] = {}
            for action in self.model_actions:
                n_comp_layer1, init_layer1, n_comp_layer2, init_layer2 = self.hyperparams[action]
                state_pi[action] = float(state_pi_raw[action])

                action_mom_payload = state_mom_raw[action]
                if not isinstance(action_mom_payload, Mapping):
                    raise TypeError(
                        f"State {state_idx}, action '{action}': action_mom payload must be a mapping."
                    )
                if "layer1" not in action_mom_payload or "layer2" not in action_mom_payload:
                    raise ValueError(
                        f"State {state_idx}, action '{action}': missing layer1/layer2 payload."
                    )

                layer1_payload = action_mom_payload["layer1"]
                layer2_payload = action_mom_payload["layer2"]
                if not isinstance(layer1_payload, Mapping) or not isinstance(layer2_payload, Sequence):
                    raise TypeError(
                        f"State {state_idx}, action '{action}': invalid layer payload types."
                    )
                if len(layer2_payload) != n_comp_layer1:
                    raise ValueError(
                        f"State {state_idx}, action '{action}': layer2 list must have {n_comp_layer1} entries."
                    )

                layer1_weights = np.asarray(layer1_payload["weights"], dtype=float)
                layer1_components_payload = layer1_payload["components"]
                if not isinstance(layer1_components_payload, Sequence):
                    raise TypeError(
                        f"State {state_idx}, action '{action}': layer1 components must be a sequence."
                    )
                if len(layer1_components_payload) != n_comp_layer1:
                    raise ValueError(
                        f"State {state_idx}, action '{action}': layer1 components length mismatch."
                    )
                layer1_components = []
                for component_payload in layer1_components_payload:
                    if not isinstance(component_payload, Mapping):
                        raise TypeError(
                            f"State {state_idx}, action '{action}': invalid layer1 component payload."
                        )
                    mean = np.asarray(component_payload["mean"], dtype=float)
                    covariance = np.asarray(component_payload["covariance"], dtype=float)
                    layer1_components.append(
                        MultivariateGaussian(mean=mean, covariance=covariance)
                    )
                layer1_mixture = MixtureModel(
                    layer1_components,
                    weights=layer1_weights,
                    init=init_layer1,
                )

                layer2_mixtures = []
                for l1_comp, layer2_mix_payload in enumerate(layer2_payload):
                    if not isinstance(layer2_mix_payload, Mapping):
                        raise TypeError(
                            f"State {state_idx}, action '{action}', layer1 {l1_comp}: invalid layer2 payload."
                        )
                    layer2_weights = np.asarray(layer2_mix_payload["weights"], dtype=float)
                    vm_components_payload = layer2_mix_payload["components"]
                    if not isinstance(vm_components_payload, Sequence):
                        raise TypeError(
                            f"State {state_idx}, action '{action}', layer1 {l1_comp}: components must be a sequence."
                        )
                    if len(vm_components_payload) != n_comp_layer2:
                        raise ValueError(
                            f"State {state_idx}, action '{action}', layer1 {l1_comp}: layer2 components length mismatch."
                        )
                    vm_components = []
                    for vm_payload in vm_components_payload:
                        if not isinstance(vm_payload, Mapping):
                            raise TypeError(
                                f"State {state_idx}, action '{action}', layer1 {l1_comp}: invalid VonMises payload."
                            )
                        vm_components.append(
                            VonMises(
                                loc=float(vm_payload["loc"]),
                                kappa=float(vm_payload["kappa"]),
                            )
                        )
                    layer2_mixtures.append(
                        MixtureModel(
                            vm_components,
                            weights=layer2_weights,
                            init=init_layer2,
                        )
                    )

                state_mom[action] = TwoLayerMoM(
                    layer1_mixture=layer1_mixture,
                    layer2_mixtures=layer2_mixtures,
                )

            action_pi_list.append(state_pi)
            action_mom_list.append(state_mom)

        self.action_pi_ = action_pi_list
        self.action_mom_ = action_mom_list

        try:
            self.check()
        except RuntimeError:
            # Emission is not bound yet; validation will run once bound by the HMM.
            pass

    def check(self) -> None:
        n_states, n_features = self._require_binding()
        if n_features != 5:
            raise ValueError(f"TwoLayerEmission expects n_features=5, got {n_features}.")

        if not isinstance(self.hyperparams, dict) or len(self.hyperparams) == 0:
            raise ValueError("hyperparams must be a non-empty dict.")

        allowed_init = {"k-means++", "k-means", "random_from_data", "random"}
        expected_actions = self.model_actions
        expected_action_set = set(expected_actions)
        action_spec: dict[str, tuple[int, int]] = {}

        for action in expected_actions:
            value = self.hyperparams[action]
            if not self.ignore_actions and action not in self.ACTION_MAP:
                raise ValueError(f"Unknown action '{action}' in hyperparams.")
            if not isinstance(value, (tuple, list)) or len(value) != 4:
                raise ValueError(
                    f"hyperparams['{action}'] must be [n_comp_layer1, init1, n_comp_layer2, init2]."
                )
            n_comp_layer1, init_layer1, n_comp_layer2, init_layer2 = value
            if not isinstance(n_comp_layer1, (int, np.integer)) or int(n_comp_layer1) < 1:
                raise ValueError(
                    f"hyperparams['{action}'][0] must be an integer >= 1, got {n_comp_layer1!r}."
                )
            if not isinstance(n_comp_layer2, (int, np.integer)) or int(n_comp_layer2) < 1:
                raise ValueError(
                    f"hyperparams['{action}'][2] must be an integer >= 1, got {n_comp_layer2!r}."
                )
            if init_layer1 not in allowed_init:
                raise ValueError(
                    f"hyperparams['{action}'][1] must be one of {sorted(allowed_init)}, got {init_layer1!r}."
                )
            if init_layer2 not in allowed_init:
                raise ValueError(
                    f"hyperparams['{action}'][3] must be one of {sorted(allowed_init)}, got {init_layer2!r}."
                )
            action_spec[action] = (int(n_comp_layer1), int(n_comp_layer2))

        if len(self.action_pi_) != n_states:
            raise ValueError(
                f"action_pi_ must contain one dict per state ({n_states}), got {len(self.action_pi_)}."
            )
        if len(self.action_mom_) != n_states:
            raise ValueError(
                f"action_mom_ must contain one dict per state ({n_states}), got {len(self.action_mom_)}."
            )

        for state in range(n_states):
            action_pi_dict = self.action_pi_[state]
            action_mom_dict = self.action_mom_[state]

            if not isinstance(action_pi_dict, dict):
                raise TypeError(f"action_pi_[{state}] must be a dict.")
            if not isinstance(action_mom_dict, dict):
                raise TypeError(f"action_mom_[{state}] must be a dict.")

            if set(action_pi_dict.keys()) != expected_action_set:
                raise ValueError(
                    f"action_pi_[{state}] keys mismatch hyperparams keys."
                )
            if set(action_mom_dict.keys()) != expected_action_set:
                raise ValueError(
                    f"action_mom_[{state}] keys mismatch hyperparams keys."
                )

            pi_values = np.array([action_pi_dict[action] for action in expected_actions], dtype=float)
            if not np.all(np.isfinite(pi_values)):
                raise ValueError(f"action_pi_[{state}] contains non-finite values.")
            if np.any(pi_values <= 0.0):
                raise ValueError(f"action_pi_[{state}] values must be strictly positive.")
            if not np.isclose(float(pi_values.sum()), 1.0, atol=1e-6):
                raise ValueError(
                    f"action_pi_[{state}] must sum to 1.0, got {float(pi_values.sum()):.12f}."
                )

            for action in expected_actions:
                n_comp_layer1, n_comp_layer2 = action_spec[action]
                mom = action_mom_dict[action]
                if not isinstance(mom, TwoLayerMoM):
                    raise TypeError(
                        f"action_mom_[{state}]['{action}'] must be a TwoLayerMoM instance."
                    )

                loc_mixture = mom.layer1_mixture
                if not isinstance(loc_mixture, MixtureModel):
                    raise TypeError(
                        f"action_mom_[{state}]['{action}'].loc_mixture must be a MixtureModel."
                    )
                if loc_mixture.n_components != n_comp_layer1:
                    raise ValueError(
                        f"State {state}, action '{action}': loc_mixture.n_components="
                        f"{loc_mixture.n_components}, expected {n_comp_layer1}."
                    )

                loc_weights = np.asarray(loc_mixture.weights, dtype=float)
                if loc_weights.shape != (n_comp_layer1,):
                    raise ValueError(
                        f"State {state}, action '{action}': loc_mixture.weights shape "
                        f"{loc_weights.shape}, expected {(n_comp_layer1,)}."
                    )
                if not np.all(np.isfinite(loc_weights)):
                    raise ValueError(
                        f"State {state}, action '{action}': loc_mixture.weights contains non-finite values."
                    )
                if np.any(loc_weights <= 0.0):
                    raise ValueError(
                        f"State {state}, action '{action}': loc_mixture.weights must be strictly positive."
                    )
                if not np.isclose(float(loc_weights.sum()), 1.0, atol=1e-6):
                    raise ValueError(
                        f"State {state}, action '{action}': loc_mixture.weights must sum to 1.0."
                    )

                if len(loc_mixture.components) != n_comp_layer1:
                    raise ValueError(
                        f"State {state}, action '{action}': loc_mixture.components length "
                        f"{len(loc_mixture.components)}, expected {n_comp_layer1}."
                    )
                for l1_comp, comp in enumerate(loc_mixture.components):
                    if not isinstance(comp, MultivariateGaussian):
                        raise TypeError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: "
                            f"component must be MultivariateGaussian."
                        )
                    mean, cov = comp.params
                    mean = np.asarray(mean, dtype=float)
                    cov = np.asarray(cov, dtype=float)
                    if mean.shape != (2,):
                        raise ValueError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: "
                            f"mean shape must be (2,), got {mean.shape}."
                        )
                    if cov.shape != (2, 2):
                        raise ValueError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: "
                            f"cov shape must be (2,2), got {cov.shape}."
                        )
                    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
                        raise ValueError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: "
                            f"mean/cov contains non-finite values."
                        )
                    if not np.allclose(cov, cov.T, atol=1e-10):
                        raise ValueError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: covariance must be symmetric."
                        )
                    eigvals = np.linalg.eigvalsh(cov)
                    if np.any(eigvals <= 0.0):
                        raise ValueError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: covariance must be positive-definite."
                        )

                if len(mom.layer2_mixtures) != n_comp_layer1:
                    raise ValueError(
                        f"State {state}, action '{action}': dir_mixtures length "
                        f"{len(mom.layer2_mixtures)}, expected {n_comp_layer1}."
                    )
                for l1_comp, dir_mixture in enumerate(mom.layer2_mixtures):
                    if not isinstance(dir_mixture, MixtureModel):
                        raise TypeError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: dir_mixture must be MixtureModel."
                        )
                    if dir_mixture.n_components != n_comp_layer2:
                        raise ValueError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: "
                            f"dir_mixture.n_components={dir_mixture.n_components}, expected {n_comp_layer2}."
                        )

                    dir_weights = np.asarray(dir_mixture.weights, dtype=float)
                    if dir_weights.shape != (n_comp_layer2,):
                        raise ValueError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: "
                            f"dir_mixture.weights shape {dir_weights.shape}, expected {(n_comp_layer2,)}."
                        )
                    if not np.all(np.isfinite(dir_weights)):
                        raise ValueError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: "
                            f"dir_mixture.weights contains non-finite values."
                        )
                    if np.any(dir_weights <= 0.0):
                        raise ValueError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: "
                            f"dir_mixture.weights must be strictly positive."
                        )
                    if not np.isclose(float(dir_weights.sum()), 1.0, atol=1e-6):
                        raise ValueError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: "
                            f"dir_mixture.weights must sum to 1.0."
                        )

                    if len(dir_mixture.components) != n_comp_layer2:
                        raise ValueError(
                            f"State {state}, action '{action}', layer1 {l1_comp}: "
                            f"dir_mixture.components length {len(dir_mixture.components)}, expected {n_comp_layer2}."
                        )
                    for l2_comp, vm in enumerate(dir_mixture.components):
                        if not isinstance(vm, VonMises):
                            raise TypeError(
                                f"State {state}, action '{action}', layer1 {l1_comp}, layer2 {l2_comp}: "
                                f"component must be VonMises."
                            )
                        loc, kappa = vm.params
                        if not np.isfinite(loc) or not np.isfinite(kappa):
                            raise ValueError(
                                f"State {state}, action '{action}', layer1 {l1_comp}, layer2 {l2_comp}: "
                                f"VonMises params must be finite."
                            )
                        if kappa <= 0.0:
                            raise ValueError(
                                f"State {state}, action '{action}', layer1 {l1_comp}, layer2 {l2_comp}: "
                                f"VonMises kappa must be > 0."
                            )

    def compute_log_likelihood(self, X: Array) -> Array:
        """
        Emission data log likelihood given state.
        """
        n_states, _ = self._require_binding()
        ll = np.zeros((X.shape[0],n_states))
        for state in range(n_states):
            for action in self.model_actions:
                indices = self._indices_for_action(X, action)
                if indices.size == 0:
                    # In multi-sequence training a sequence can miss one or more actions.
                    # Skip absent actions for this sequence chunk.
                    continue
                X_action = X[indices]
                ll[indices, state] = (
                    np.log(self.action_pi_[state][action]) #scalar
                    + self.action_mom_[state][action].log_pdf(X_action[:, 1:3], X_action[:,3:5]) #vector
                )

        return ll

    def initialize_sufficient_statistics(self) -> dict[str, Any]:
        n_states, n_features = self._require_binding()
        return {
            "hmm_post_sum": np.zeros((n_states,), dtype=float),

            "action_post": {action: np.zeros((n_states,), dtype=float)
                            for action in self.model_actions},

            "mix_layer1_post": {action: np.zeros((n_states, self.hyperparams[action][0]),
                                                 dtype=float)
                            for action in self.model_actions},

            "mix_layer1_gauss": {action: np.zeros((n_states, self.hyperparams[action][0], 2+2**2),
                                                dtype=float)
                            for action in self.model_actions},

            "mix_layer2_post": {action: np.zeros((n_states, self.hyperparams[action][0], self.hyperparams[action][2]),
                                                 dtype=float)
                            for action in self.model_actions},

            "mix_layer2_vm": {action: np.zeros((n_states, self.hyperparams[action][0], self.hyperparams[action][2], 2),
                                                dtype=float)
                            for action in self.model_actions},
        }

    def accumulate_sufficient_statistics(
        self,
        stats: dict[str, Any],
        X: Array,
        posteriors: Array, #gamma hmm posterior
        params: str,
    ) -> None:
        """
        Update sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by self.initialize_sufficient_statistics`.

        X : array, shape (n_samples, n_features)
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

        X: Array shape (n_obs, 5)
        Observable variable x_n is a tuple of (a_n, x-axis_n, y-axis_n, cos_n, sin_n).
        """
        n_states, n_features = self._require_binding()
        posteriors = np.asarray(posteriors, dtype=float) #(n_obs, n_states)
        sum_hmm_post = posteriors.sum(axis=0)

        if n_features != 5:
            raise ValueError("TwoLayerEmission must have 5 features.")
        if np.any(sum_hmm_post <= 0.0):
            warnings.warn(
                "Sum of gamma is not positive; skipping this batch in stats accumulation.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        stats['hmm_post_sum'] += sum_hmm_post

        for action in self.model_actions:
            n_comp_layer1 = self.hyperparams[action][0]
            n_comp_layer2 = self.hyperparams[action][2]
            indices = self._indices_for_action(X, action)
            n_obs_in_action = len(indices)
            if n_obs_in_action == 0:
                # Missing actions are expected for sequence-level chunks when using lengths.
                continue
            X_action = X[indices]

            resp_layer1_in_action = np.zeros((n_obs_in_action, n_states, n_comp_layer1),
                                             dtype=float)
            resp_layer2_in_action = np.zeros((n_obs_in_action, n_states, n_comp_layer1, n_comp_layer2),
                                             dtype=float)

            for state in range(n_states):
                resp_layer1_in_action[:, state, :] = (self.action_mom_[state][action].
                                                      l1_responsibilities(X_action[:,1:3], X_action[:,3:5]))
                resp_layer2_in_action[:, state, :, :] = (self.action_mom_[state][action].
                                                         l2_responsibilities(X_action[:,3:5]))

            # accumulate sufficient statistics
            # indices. n: sample_in_action; i: state; j: layer 1 component; k: layer 2 component. f: feature.

            stats["action_post"][action] += posteriors[indices].sum(axis=0)
            stats["mix_layer1_post"][action] += np.einsum(
                "ni,nij->ij",
                posteriors[indices],  # shape (n_obs_in_action, n_states)
                resp_layer1_in_action  # shape (n_obs_in_action, n_states, n_comp_layer1)
            )
            stats["mix_layer1_gauss"][action] += np.einsum(
                "ni,nij,nf->ijf",
                posteriors[indices],
                resp_layer1_in_action,
                self.get_gaussian_sufficient_stat(X[indices, 1:3])
            )
            stats["mix_layer2_post"][action] += np.einsum(
                "ni,nij,nijk->ijk",
                posteriors[indices],
                resp_layer1_in_action,
                resp_layer2_in_action
            )
            stats["mix_layer2_vm"][action] += np.einsum(
                "ni,nij,nijk,nf->ijkf",
                posteriors[indices],
                resp_layer1_in_action,
                resp_layer2_in_action,
                X[indices,3:5]
            )


    def m_step(self, stats: dict[str, Any], params: str) -> None:
        n_states, n_features = self._require_binding()
        if "a" not in params and "m" not in params:
            return
        #self.check()
        for action in self.model_actions:
            n_comp_layer1 = self.hyperparams[action][0]
            n_comp_layer2 = self.hyperparams[action][2]
            for state in range(n_states):
                # update action prob (Categorical dist)
                if "a" in params:
                    action_pi = float(stats["action_post"][action][state]/
                                        stats['hmm_post_sum'][state])
                    self.action_pi_[state][action] = action_pi
                if "m" in params:
                    # update 1st layer Mixture
                    layer1_pi = (stats["mix_layer1_post"][action][state,:]
                                        / stats["action_post"][action][state])
                    self.action_mom_[state][action].layer1_mixture.weights = layer1_pi #setter normalize internally
                    for l1_comp in range(n_comp_layer1):
                        gauss = (stats["mix_layer1_gauss"][action][state,l1_comp,:]
                                    / stats["mix_layer1_post"][action][state,l1_comp])
                        self.action_mom_[state][action].layer1_mixture.components[l1_comp].dual_param = gauss
                        # update 2nd layer Mixture
                        layer2_pi = (stats["mix_layer2_post"][action][state,l1_comp,:]
                                        / stats["mix_layer1_post"][action][state,l1_comp])
                        self.action_mom_[state][action].layer2_mixtures[l1_comp].weights = layer2_pi
                        for l2_comp in range(n_comp_layer2):
                            vm = (stats["mix_layer2_vm"][action][state,l1_comp,l2_comp,:]
                                        / stats["mix_layer2_post"][action][state,l1_comp,l2_comp])
                            self.action_mom_[state][action].layer2_mixtures[l1_comp].components[l2_comp].dual_param = vm


    def sample_from_state(self, state: int, random_state: Any) -> Array:
        raise NotImplementedError("TwoLayerEmission is not implemented yet.")


class TwoLayerHMM(EmissionHMM):
    """
    Gaussian-emission HMM using the modular emission architecture.

    This class mirrors the common ``GaussianHMM`` workflow while delegating
    emission logic to :class:`GaussianEmission`.
    """

    def __init__(
        self,
        emission_hyperparams,
        n_components: int = 1,
        *,
        ignore_actions: bool = False,
        startprob_prior: Any = 1.0,
        transmat_prior: Any = 1.0,
        algorithm: str = "viterbi",
        random_state: Any = None,
        n_iter: int = 10,
        tol: float = 1e-2,
        verbose: bool = False,
        params: str = "stam",
        init_params: str = "stam",
        implementation: str = "log",
    ) -> None:

        self.emission_hyperparams = emission_hyperparams
        self.ignore_actions = bool(ignore_actions)
        emission = TwoLayerEmission(
            emission_hyperparams,
            ignore_actions=ignore_actions,
        )

        super().__init__(
            n_components=n_components,
            emission=emission,
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

    def export_model_params(self) -> dict[str, Any]:
        if not hasattr(self, "startprob_") or not hasattr(self, "transmat_"):
            raise RuntimeError(
                "Model transition parameters are not initialized. Fit the model first."
            )
        if not hasattr(self, "n_features"):
            raise RuntimeError(
                "Model n_features is not initialized. Fit the model first."
            )
        self.emission.bind(self.n_components, self.n_features)
        self.emission.check()
        return {
            "n_components": int(self.n_components),
            "n_features": int(self.n_features),
            "startprob": np.asarray(self.startprob_, dtype=float).tolist(),
            "transmat": np.asarray(self.transmat_, dtype=float).tolist(),
            "emission": self.emission.export_params(),
        }

    def set_model_params(self, params: Mapping[str, Any]) -> None:
        if not isinstance(params, Mapping):
            raise TypeError(
                "params must be a mapping produced by export_model_params()."
            )
        required_keys = {"n_components", "n_features", "startprob", "transmat", "emission"}
        missing = sorted(required_keys - set(params.keys()))
        if missing:
            raise ValueError(f"Missing keys in model params: {missing}.")

        emission_params = params["emission"]
        if not isinstance(emission_params, Mapping):
            raise TypeError("params['emission'] must be a mapping.")
        if "hyperparams" not in emission_params:
            raise ValueError("params['emission'] must include 'hyperparams'.")

        n_components = int(params["n_components"])
        n_features = int(params["n_features"])
        self.n_components = n_components
        self.n_features = n_features
        startprob = np.asarray(params["startprob"], dtype=float)
        if startprob.shape != (n_components,):
            raise ValueError(
                f"params['startprob'] must have shape {(n_components,)}, got {startprob.shape}."
            )
        if not np.all(np.isfinite(startprob)):
            raise ValueError("params['startprob'] contains non-finite values.")
        if np.any(startprob < 0.0):
            raise ValueError("params['startprob'] must be non-negative.")
        startprob_sum = float(startprob.sum())
        if startprob_sum <= 0.0:
            raise ValueError("params['startprob'] must sum to a positive value.")
        self.startprob_ = startprob / startprob_sum

        transmat = np.asarray(params["transmat"], dtype=float)
        if transmat.shape != (n_components, n_components):
            raise ValueError(
                "params['transmat'] must have shape "
                f"{(n_components, n_components)}, got {transmat.shape}."
            )
        if not np.all(np.isfinite(transmat)):
            raise ValueError("params['transmat'] contains non-finite values.")
        if np.any(transmat < 0.0):
            raise ValueError("params['transmat'] must be non-negative.")
        row_sums = transmat.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError(
                "Each row of params['transmat'] must sum to a positive value."
            )
        self.transmat_ = transmat / row_sums

        ignore_actions = bool(emission_params.get("ignore_actions", self.ignore_actions))
        hyperparams = emission_params["hyperparams"]
        self.ignore_actions = ignore_actions
        self.emission_hyperparams = hyperparams
        self.emission = TwoLayerEmission(
            hyperparams,
            ignore_actions=ignore_actions,
        )
        self.emission.bind(self.n_components, self.n_features)
        self.emission.set_emission_params(emission_params)
        self._check()
