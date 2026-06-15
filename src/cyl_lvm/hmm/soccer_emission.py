"""Soccer emission with optional fixed action-specific MoM initialization."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..core.types import Array
from ..distributions import MultivariateGaussian, VonMises
from ..hierarchical import TwoLayerMoM
from ..mixtures import MixtureModel
from .hmm_two_layer import TwoLayerEmission


class SoccerEmission(TwoLayerEmission):
    """
    Multi-action two-layer emission with fixed MoM initialization for selected actions.

    By default, actions ``starting`` and ``pass`` are initialized with hard-coded
    MoM parameters
    (3 hidden states, 6 Gaussian components in layer-1, and 2 VonMises components
    in layer-2). Other actions are initialized with the regular data-driven logic
    from :class:`TwoLayerEmission`.
    """

    _STARTING_ACTION_NAME = "starting"
    _PASS_ACTION_NAME = "pass"
    _STARTING_STATE_GAUSSIAN_PARAMS = (
        [  # State 0: Down field
            ([18.0, 18.0], [[80.0, 0.0], [0.0, 80.0]]),
            ([45.0, 48.0], [[80.0, 0.0], [0.0, 80.0]]),
            ([45.0, 18.0], [[80.0, 0.0], [0.0, 80.0]]),
            ([18.0, 48.0], [[80.0, 0.0], [0.0, 80.0]]),
            ([30.0, 1.0], [[200.0, 0.0], [0.0, 1.2]]),
            ([30.0, 67.0], [[200.0, 0.0], [0.0, 1.2]]),
        ],
        [  # State 1: Mid field
            ([52.0, 18.0], [[80.0, 0.0], [0.0, 80.0]]),
            ([52.0, 48.0], [[80.0, 0.0], [0.0, 80.0]]),
            ([65.0, 1.0], [[180.0, 0.0], [0.0, 1.2]]),
            ([65.0, 67.0], [[180.0, 0.0], [0.0, 1.2]]),
            ([35.0, 1.0], [[180.0, 0.0], [0.0, 1.2]]),
            ([35.0, 67.0], [[180.0, 0.0], [0.0, 1.2]]),
        ],
        [  # State 2: Up field
            ([71.0, 18.0], [[80.0, 0.0], [0.0, 80.0]]),
            ([71.0, 48.0], [[80.0, 0.0], [0.0, 80.0]]),
            ([78.0, 1.0], [[180.0, 0.0], [0.0, 1.2]]),
            ([78.0, 67.0], [[180.0, 0.0], [0.0, 1.2]]),
            ([103.5, 1.0], [[4.0, 0.0], [0.0, 4.0]]),
            ([103.5, 67.0], [[4.0, 0.0], [0.0, 4.0]]),
        ],
    )
    _STARTING_VON_MISES_COMPONENTS = (
        [  # State 0: Down field
            ((0.0, 2.0), (2.2, 2.0)),
            ((-0.5, 2.0), (-2.5, 2.0)),
            ((0.5, 2.0), (2.5, 2.0)),
            ((-0.0, 2.0), (-2.2, 2.0)),
            ((2.3, 2.0), (0.7, 2.0)),
            ((-2.3, 2.0), (-0.7, 2.0)),
        ],
        [  # State 1: Mid field
            ((0.3, 2.0), (1.8, 2.0)),
            ((-0.3, 2.0), (-1.8, 2.0)),
            ((0.5, 2.0), (2.5, 2.0)),
            ((-0.5, 2.0), (-2.5, 2.0)),
            ((2.8, 2.0), (1.0, 2.0)),
            ((-2.8, 2.0), (-1.0, 2.0)),
        ],
        [  # State 2: Up field
            ((0.3, 2.0), (1.5, 2.0)),
            ((-0.3, 2.0), (-1.5, 2.0)),
            ((0.5, 2.0), (2.5, 2.0)),
            ((-0.5, 2.0), (-2.5, 2.0)),
            ((2.8, 2.0), (1.8, 2.0)),
            ((-2.8, 2.0), (-1.8, 2.0)),
        ],
    )
    _PASS_STATE_GAUSSIAN_PARAMS = (
        [  # State 0
            ([18, 18], [[80, 0], [0, 80]]),
            ([38, 55], [[80, 0.0], [0.0, 80]]),
            ([38, 10], [[80, 0], [0, 80]]),
            ([18, 48], [[80, 0.0], [0.0, 80]]),
        ],
        [  # State 1
            ([52, 20], [[80, 0], [0, 80]]),
            ([52, 50], [[80, 0.0], [0.0, 80]]),
            ([65, 4], [[120, 0.0], [0.0, 3.2]]),
            ([65, 64], [[120, 0.0], [0.0, 3.2]]),
        ],
        [  # State 2
            ([71, 18], [[80, 0], [0, 80]]),
            ([71, 48], [[80, 0.0], [0.0, 80]]),
            ([78, 2.5], [[130, 0.0], [0.0, 4]]),
            ([78, 66.5], [[130, 0.0], [0.0, 4]]),
        ],
    )
    _PASS_VON_MISES_COMPONENTS = (
        [  # State 0
            ((0.0, 2.0), (2.2, 2.0)),
            ((-0.5, 2.0), (-2.5, 2.0)),
            ((0.5, 2.0), (2.5, 2.0)),
            ((-0.0, 2.0), (-2.2, 2.0)),
        ],
        [  # State 1
            ((0.3, 2.0), (1.8, 2.0)),
            ((-0.3, 2.0), (-1.8, 2.0)),
            ((0.5, 2.0), (2.5, 2.0)),
            ((-0.5, 2.0), (-2.5, 2.0)),
        ],
        [  # State 2
            ((0.3, 2.0), (1.5, 2.0)),
            ((-0.3, 2.0), (-1.5, 2.0)),
            ((0.5, 2.0), (2.5, 2.0)),
            ((-0.5, 2.0), (-2.5, 2.0)),
        ],
    )

    _FIXED_STATE_ACTION_PI = (
        {"starting": 0.10, "pass": 0.45, "dribble": 0.45, "shot": 1e-7},
        {"starting": 0.10, "pass": 0.45, "dribble": 0.45, "shot": 1e-7},
        {"starting": 0.06, "pass": 0.42, "dribble": 0.42, "shot": 0.10},
    )

    def __init__(
            self,
            emission_hyperparams: Mapping[str, Sequence[Any]],
            *,
            fixed_action_params: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            emission_hyperparams=dict(emission_hyperparams),
            ignore_actions=False,
        )
        self.fixed_action_params: dict[str, Mapping[str, Any]] = {
            self._STARTING_ACTION_NAME: {
                "state_layer1_params": self._STARTING_STATE_GAUSSIAN_PARAMS,
                "layer2_component_params": self._STARTING_VON_MISES_COMPONENTS,
                "layer1_weights": None,
                "layer2_weights": None,
            },
            self._PASS_ACTION_NAME: {
                "state_layer1_params": self._PASS_STATE_GAUSSIAN_PARAMS,
                "layer2_component_params": self._PASS_VON_MISES_COMPONENTS,
                "layer1_weights": None,
                "layer2_weights": None,
            },
            "dribble": {
                "state_layer1_params": self._PASS_STATE_GAUSSIAN_PARAMS,
                "layer2_component_params": self._PASS_VON_MISES_COMPONENTS,
                "layer1_weights": None,
                "layer2_weights": None,
            }
        }
        if fixed_action_params:
            self.fixed_action_params.update(
                {str(action): spec for action, spec in fixed_action_params.items()}
            )

    @staticmethod
    def _build_state_mom(
            *,
            n_layer1_components: int,
            n_layer2_components: int,
            init_layer1: str,
            init_layer2: str,
            state_layer1_component_params: Sequence[tuple[Sequence[float], Sequence[Sequence[float]]]],
            layer1_weights: Sequence[float] | None,
            layer2_component_params: Sequence[Sequence[tuple[float, float]]],
            layer2_weights: Sequence[float] | None,
    ) -> TwoLayerMoM:
        mom = TwoLayerMoM(
            layer1_mixture=MixtureModel(
                [MultivariateGaussian(2) for _ in range(n_layer1_components)],
                init=init_layer1,
            ),
            layer2_mixtures=[
                MixtureModel(
                    [VonMises() for _ in range(n_layer2_components)],
                    init=init_layer2,
                )
                for _ in range(n_layer1_components)
            ],
        )

        if len(state_layer1_component_params) != n_layer1_components:
            raise ValueError(
                "state_layer1_component_params length must match n_layer1_components."
            )
        if len(layer2_component_params) != n_layer1_components:
            raise ValueError(
                "layer2_component_params length must match n_layer1_components."
            )

        gaussian_params: list[tuple[Array, Array]] = [
            (np.asarray(mean, dtype=float), np.asarray(cov, dtype=float))
            for mean, cov in state_layer1_component_params
        ]
        layer1_weights_arr = (
            np.ones(n_layer1_components, dtype=float)
            if layer1_weights is None
            else np.asarray(layer1_weights, dtype=float)
        )
        mom.layer1_mixture.set_params(
            component_params=gaussian_params,
            weights=layer1_weights_arr,
        )

        layer2_weights_arr = (
            np.ones(n_layer2_components, dtype=float)
            if layer2_weights is None
            else np.asarray(layer2_weights, dtype=float)
        )
        for l1_comp, component_params in enumerate(layer2_component_params):
            if len(component_params) != n_layer2_components:
                raise ValueError(
                    "Each layer-1 component in layer2_component_params must define "
                    "n_layer2_components entries."
                )
            layer2_component_params_tuple = [
                (float(loc), float(kappa)) for loc, kappa in component_params
            ]
            mom.layer2_mixtures[l1_comp].set_params(
                component_params=layer2_component_params_tuple,
                weights=layer2_weights_arr,
            )

        return mom

    def initialize(self, X: Array, init_params: str, random_state: Any) -> None:
        """
        Initialize all actions with base logic, then overwrite fixed-action MoM params.
        """
        super().initialize(X, init_params, random_state)

        n_states, n_features = self._require_binding()
        if n_features != 5:
            raise ValueError(f"SoccerEmission expects n_features=5, got {n_features}.")

        for action, action_spec in self.fixed_action_params.items():
            if action not in self.model_actions:
                raise ValueError(
                    f"Fixed action '{action}' is not present in emission_hyperparams."
                )
            if not isinstance(action_spec, Mapping):
                raise TypeError(f"Fixed action spec for '{action}' must be a mapping.")
            if "state_layer1_params" not in action_spec or "layer2_component_params" not in action_spec:
                raise ValueError(
                    f"Fixed action spec for '{action}' must include "
                    "'state_layer1_params' and 'layer2_component_params'."
                )

            state_layer1_params = action_spec["state_layer1_params"]
            layer2_component_params = action_spec["layer2_component_params"]
            layer1_weights = action_spec.get("layer1_weights", None)
            layer2_weights = action_spec.get("layer2_weights", None)

            if len(state_layer1_params) != n_states:
                raise ValueError(
                    f"Fixed action '{action}' defines {len(state_layer1_params)} states, "
                    f"but model has {n_states}."
                )
            if len(layer2_component_params) != n_states:
                raise ValueError(
                    f"Fixed action '{action}' defines {len(layer2_component_params)} "
                    f"layer2 state blocks, but model has {n_states}."
                )

            n_layer1_components, init_layer1, n_layer2_components, init_layer2 = self.hyperparams[action]
            for state in range(n_states):
                self.action_mom_[state][action] = self._build_state_mom(
                    n_layer1_components=int(n_layer1_components),
                    n_layer2_components=int(n_layer2_components),
                    init_layer1=str(init_layer1),
                    init_layer2=str(init_layer2),
                    state_layer1_component_params=state_layer1_params[state],
                    layer1_weights=layer1_weights,
                    layer2_component_params=layer2_component_params[state],
                    layer2_weights=layer2_weights,
                )

        if len(self._FIXED_STATE_ACTION_PI) != n_states:
            raise ValueError(
                "SoccerEmission fixed action priors are hard-coded for "
                f"{len(self._FIXED_STATE_ACTION_PI)} states, got n_components={n_states}."
            )

        expected_actions = tuple(self.model_actions)
        expected_action_set = set(expected_actions)
        for state in range(n_states):
            state_pi_raw = self._FIXED_STATE_ACTION_PI[state]
            if set(state_pi_raw.keys()) != expected_action_set:
                raise ValueError(
                    f"Fixed action priors for state {state} must match actions "
                    f"{sorted(expected_action_set)}."
                )
            pi_values = np.array(
                [float(state_pi_raw[action]) for action in expected_actions],
                dtype=float,
            )
            if not np.all(np.isfinite(pi_values)):
                raise ValueError(
                    f"Fixed action priors for state {state} contain non-finite values."
                )
            if np.any(pi_values <= 0.0):
                raise ValueError(
                    f"Fixed action priors for state {state} must be strictly positive."
                )
            pi_values /= float(pi_values.sum())
            self.action_pi_[state] = {
                action: float(pi_values[idx])
                for idx, action in enumerate(expected_actions)
            }
