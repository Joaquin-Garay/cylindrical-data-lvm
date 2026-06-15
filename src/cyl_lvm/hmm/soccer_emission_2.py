"""One-action soccer emission with fixed MoM initialization."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..core.types import Array
from ..distributions import MultivariateGaussian, VonMises
from ..hierarchical import TwoLayerMoM
from ..mixtures import MixtureModel
from .hmm_two_layer import TwoLayerEmission


class SoccerEmission2(TwoLayerEmission):
    """
    One-action two-layer emission with fixed MoM initialization.

    This class forces ``ignore_actions=True`` and expects exactly one action in
    ``emission_hyperparams``. The action is initialized from hard-coded MoM
    parameters (6 hidden states, 6 Gaussian layer-1 components, and 2 VonMises
    layer-2 components).
    """

    _STATE_GAUSSIAN_PARAMS = (
        [  # State 0
            ([10, 22], [[50, 0], [0, 50]]),
            ([28, 13], [[60, 0.0], [0.0, 60]]),
            ([38, 7], [[80, 0], [0, 40]]),
        ],
        [  # State 1
            ([10, 46], [[50, 0], [0, 50]]),
            ([28, 55], [[60, 0.0], [0.0, 60]]),
            ([38, 61], [[80, 0], [0, 40]]),
        ],
        [  # State 2
            ([52, 34], [[30, 0], [0, 180]]),
            ([54, 1], [[150, 0.0], [0.0, 3.2]]),
            ([54, 67], [[150, 0.0], [0.0, 3.2]]),
        ],
        [  # State 3
            ([100, 18], [[80, 0], [0, 80]]),
            ([90, 10], [[80, 0.0], [0.0, 80]]),
            ([78, 1], [[180, 0.0], [0.0, 4]]),
        ],
        [  # State 4
            ([100, 50], [[80, 0], [0, 80]]),
            ([90, 58], [[80, 0.0], [0.0, 80]]),
            ([78, 67], [[180, 0.0], [0.0, 4]]),
        ],
        [  # State 5
            ([90, 44], [[80, 0], [0, 80]]),
            ([90, 24], [[80, 0.0], [0.0, 80]]),
            ([70, 34], [[80, 0.0], [0.0, 80]]),
        ],
    )

    _VON_MISES_COMPONENTS = (
        [  # State 0
            ((0.0, 2.0), (2.2, 2.0)),
            ((0.5, 2.0), (2.5, 2.0)),
            ((0.5, 2.0), (2.5, 2.0)),
        ],
        [  # State 1
            ((-0.0, 2.0), (-2.2, 2.0)),
            ((-0.5, 2.0), (-2.5, 2.0)),
            ((-0.5, 2.0), (-2.5, 2.0)),
        ],
        [  # State 2
            ((1.5, 2.0), (-1.5, 2.0)),
            ((0.5, 2.0), (2.5, 2.0)),
            ((-0.5, 2.0), (-2.5, 2.0)),
        ],
        [  # State 3
            ((0.3, 2.0), (2.9, 2.0)),
            ((0.3, 2.0), (2.9, 2.0)),
            ((0.5, 2.0), (2.5, 2.0)),
        ],
        [  # State 4
            ((-0.3, 2.0), (-2.9, 2.0)),
            ((-0.3, 2.0), (-2.9, 2.0)),
            ((-0.5, 2.0), (-2.5, 2.0)),
        ],
        [  # State 5
            ((-0.3, 2.0), (-2.9, 2.0)),
            ((-0.3, 2.0), (-2.9, 2.0)),
            ((-0.5, 2.0), (-2.5, 2.0)),
        ],
    )

    def __init__(
            self,
            emission_hyperparams: Mapping[str, Sequence[Any]],
            *,
            fixed_action_params: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            emission_hyperparams=dict(emission_hyperparams),
            ignore_actions=True,
        )
        default_action = self.model_actions[0]
        self.fixed_action_params: dict[str, Mapping[str, Any]] = {
            default_action: {
                "state_layer1_params": self._STATE_GAUSSIAN_PARAMS,
                "layer2_component_params": self._VON_MISES_COMPONENTS,
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
        Initialize directly from fixed-action parameters (no data-driven MoM init).
        """
        _ = init_params, random_state
        X = np.asarray(X, dtype=float)

        n_states, n_features = self._require_binding()
        if X.ndim != 2 or X.shape[1] != n_features:
            raise ValueError(
                f"Expected X with shape (n_samples, {n_features}), got {X.shape}."
            )
        if n_features != 5:
            raise ValueError(f"SoccerEmission2 expects n_features=5, got {n_features}.")

        action = self.model_actions[0]
        unexpected_actions = sorted(
            set(self.fixed_action_params.keys()) - {action}
        )
        if unexpected_actions:
            raise ValueError(
                "With ignore_actions=True, fixed_action_params must only contain "
                f"the model action '{action}'. Got unexpected keys: {unexpected_actions}."
            )
        if action not in self.fixed_action_params:
            raise ValueError(
                f"Missing fixed parameters for model action '{action}'."
            )

        action_spec = self.fixed_action_params[action]
        if not isinstance(action_spec, Mapping):
            raise TypeError(f"Fixed action spec for '{action}' must be a mapping.")
        if (
            "state_layer1_params" not in action_spec
            or "layer2_component_params" not in action_spec
        ):
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

        if self.action_pi_:
            self.action_pi_.clear()
        if self.action_mom_:
            self.action_mom_.clear()

        for state in range(n_states):
            state_mom = self._build_state_mom(
                n_layer1_components=int(n_layer1_components),
                n_layer2_components=int(n_layer2_components),
                init_layer1=str(init_layer1),
                init_layer2=str(init_layer2),
                state_layer1_component_params=state_layer1_params[state],
                layer1_weights=layer1_weights,
                layer2_component_params=layer2_component_params[state],
                layer2_weights=layer2_weights,
            )
            self.action_pi_.append({action: 1.0})
            self.action_mom_.append({action: state_mom})
