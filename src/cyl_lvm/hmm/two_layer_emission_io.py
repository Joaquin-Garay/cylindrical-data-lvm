"""I/O helpers for ``TwoLayerEmission`` parameters."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..distributions import MultivariateGaussian, VonMises
from ..hierarchical import TwoLayerMoM
from ..mixtures import MixtureModel


class TwoLayerEmissionIOMixin:
    """Mixin implementing parameter export/import/print for TwoLayerEmission."""

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

    @staticmethod
    def _fmt_vector(values: Sequence[float], decimals: int) -> str:
        return "[" + ", ".join(f"{float(v):.{decimals}f}" for v in values) + "]"

    @staticmethod
    def _fmt_matrix(values: Sequence[Sequence[float]], decimals: int) -> str:
        rows = [TwoLayerEmissionIOMixin._fmt_vector(row, decimals) for row in values]
        return "[" + ", ".join(rows) + "]"

    @staticmethod
    def _fmt_probability_vector(values: Sequence[float], decimals: int = 1) -> str:
        return "[" + ", ".join(f"{100.0 * float(v):.{decimals}f}%" for v in values) + "]"

    def print_params(
        self,
        *,
        decimals: int = 3,
        show_mom: bool = True,
        startprob: Sequence[float] | None = None,
        transmat: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """
        Pretty-print emission parameters in a human-readable format.

        startprob/transmat are optional model-level probabilities from the owning HMM.
        """
        payload = self.export_params()
        model_actions = payload["model_actions"]
        n_states = len(payload["action_pi"])

        print("TwoLayerEmission Parameters")
        print(f'Ignore actions: {payload["ignore_actions"]}')
        print("")

        if startprob is not None:
            startprob_arr = np.asarray(startprob, dtype=float)
            if startprob_arr.shape != (n_states,):
                raise ValueError(
                    f"startprob must have shape {(n_states,)}, got {startprob_arr.shape}."
                )
            if not np.all(np.isfinite(startprob_arr)):
                raise ValueError("startprob contains non-finite values.")
            total = float(startprob_arr.sum())
            if total <= 0.0 or not np.isfinite(total):
                raise ValueError("startprob must sum to a finite positive value.")
            startprob_arr = startprob_arr / total
            print("Start probabilities:")
            print("  " + self._fmt_probability_vector(startprob_arr.tolist(), 1))
            print("")

        if transmat is not None:
            transmat_arr = np.asarray(transmat, dtype=float)
            if transmat_arr.shape != (n_states, n_states):
                raise ValueError(
                    "transmat must have shape "
                    f"{(n_states, n_states)}, got {transmat_arr.shape}."
                )
            if not np.all(np.isfinite(transmat_arr)):
                raise ValueError("transmat contains non-finite values.")
            row_sums = transmat_arr.sum(axis=1, keepdims=True)
            if np.any(row_sums <= 0.0):
                raise ValueError("Each transmat row must sum to a positive value.")
            transmat_arr = transmat_arr / row_sums
            print("Transition matrix:")
            for state in range(n_states):
                row_text = self._fmt_probability_vector(transmat_arr[state].tolist(), 1)
                print(f"  From state {state}: {row_text}")
            print("")

        print("Hyperparameters:")
        for action in model_actions:
            n_comp_layer1, init_layer1, n_comp_layer2, init_layer2 = payload["hyperparams"][action]
            print(
                f'  "{action}": '
                f"layer1 components={n_comp_layer1} (init={init_layer1}), "
                f"layer2 components={n_comp_layer2} (init={init_layer2})"
            )
        print("")

        print("Action pi:")
        for state, state_pi in enumerate(payload["action_pi"]):
            print(f"  State {state}:")
            for action in model_actions:
                percentage = 100.0 * float(state_pi[action])
                print(f'    "{action}" = {percentage:.1f}%')
        print("")

        if show_mom:
            print("Action MoM:")
            for state, state_mom in enumerate(payload["action_mom"]):
                print(f"  State {state}:")
                for action in model_actions:
                    action_payload = state_mom[action]
                    layer1_payload = action_payload["layer1"]
                    layer2_payload = action_payload["layer2"]
                    layer1_weights = layer1_payload["weights"]
                    print(f'    Action "{action}":')
                    print(
                        "      Layer1 weights = "
                        + self._fmt_probability_vector(layer1_weights, 1)
                    )
                    for l1_comp, (layer1_component, layer2_mix) in enumerate(
                        zip(layer1_payload["components"], layer2_payload)
                    ):
                        mean = layer1_component["mean"]
                        covariance = layer1_component["covariance"]
                        print(
                            f"      L1 component {l1_comp}: "
                            f"mean={self._fmt_vector(mean, decimals)}, "
                            f"cov={self._fmt_matrix(covariance, decimals)}"
                        )
                        print(
                            f"        Layer2 weights = "
                            f"{self._fmt_probability_vector(layer2_mix['weights'], 1)}"
                        )
                        vm_text = ", ".join(
                            (
                                f"(loc={float(vm['loc']):.{decimals}f}, "
                                f"kappa={float(vm['kappa']):.{decimals}f})"
                            )
                            for vm in layer2_mix["components"]
                        )
                        print(f"        Layer2 components = [{vm_text}]")
                print("")

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

            raw_pi_values = np.asarray(
                [float(state_pi_raw[action]) for action in self.model_actions],
                dtype=float,
            )
            if not np.all(np.isfinite(raw_pi_values)):
                raise ValueError(
                    f"State {state_idx}: action_pi contains non-finite values."
                )
            if np.any(raw_pi_values <= 0.0):
                raise ValueError(
                    f"State {state_idx}: action_pi values must be strictly positive."
                )
            pi_total = float(raw_pi_values.sum())
            if not np.isfinite(pi_total) or pi_total <= 0.0:
                raise ValueError(
                    f"State {state_idx}: action_pi values must sum to a finite positive number."
                )
            normalized_pi = raw_pi_values / pi_total
            state_pi: dict[str, float] = {
                action: float(value)
                for action, value in zip(self.model_actions, normalized_pi)
            }
            state_mom: dict[str, TwoLayerMoM] = {}
            for action in self.model_actions:
                n_comp_layer1, init_layer1, n_comp_layer2, init_layer2 = self.hyperparams[action]

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
                        MultivariateGaussian(int(mean.shape[0]), mean=mean, covariance=covariance)
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
