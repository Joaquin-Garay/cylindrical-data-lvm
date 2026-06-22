"""Feature engineering helpers for SPADL actions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..core.types import Array

__all__ = ["consolidate", "add_noise", "prepare_data"]

_FEATURE_COLUMNS = ["action_type_id", "start_x", "start_y", "cos_angle", "sin_angle"]
_BASE_REQUIRED_COLUMNS = {
    "type_name",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "game_id",
    "period_id",
    "team_id",
}


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}.")


def _empty_features() -> tuple[Array, Array]:
    return np.empty((0, len(_FEATURE_COLUMNS)), dtype=float), np.empty(0, dtype=int)


def _compute_new_possession_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    return (
        (df.index == df.index[0])
        | (df["game_id"] != df["game_id"].shift(1))
        | (df["period_id"] != df["period_id"].shift(1))
        | (df["team_id"] != df["team_id"].shift(1))
        | (df["type_name"] == "goalkick")
        | (df["type_name"] == "freekick")
        | (df["type_name"] == "throw_in")
        | (df["type_name"] == "corner")
        | (df["type_name"].shift(1) == "shot")
        | (df["type_name"].shift(1) == "bad_touch")
        | (df["type_name"].shift(1) == "foul")
    )


def consolidate(actions: pd.DataFrame) -> pd.DataFrame:
    """Merge related action labels and normalize penalty start location."""
    _require_columns(actions, {"type_name", "start_x", "start_y"})
    df = actions.copy()

    corner_idx = df["type_name"].str.contains("corner", na=False)
    df["type_name"] = df["type_name"].mask(corner_idx, "corner")

    freekick_idx = df["type_name"].str.contains("freekick", na=False)
    df["type_name"] = df["type_name"].mask(freekick_idx, "freekick")

    keeper_idx = df["type_name"].str.contains("keeper", na=False)
    df["type_name"] = df["type_name"].mask(keeper_idx, "keeper_action")

    df["start_x"] = df["start_x"].mask(df["type_name"] == "shot_penalty", 94.5)
    df["start_y"] = df["start_y"].mask(df["type_name"] == "shot_penalty", 34.0)
    return df


def add_noise(
    actions: pd.DataFrame,
    *,
    sigma: float = 0.5,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add Gaussian noise to selected start/end action coordinates."""
    _require_columns(actions, {"type_name", "start_x", "start_y", "end_x", "end_y"})
    df = actions.copy()

    rng = np.random.default_rng(random_state)

    start_list = [
        "cross",
        "shot",
        "dribble",
        "pass",
        "keeper_action",
        "clearance",
        "goalkick",
    ]
    start_mask = df["type_name"].isin(start_list)
    if start_mask.any():
        start_noise = rng.normal(
            0.0, sigma, size=df.loc[start_mask, ["start_x", "start_y"]].shape
        )
        df.loc[start_mask, ["start_x", "start_y"]] += start_noise

    end_list = [
        "cross",
        "shot",
        "dribble",
        "pass",
        "keeper_action",
        "throw_in",
        "corner",
        "freekick",
        "shot_penalty",
    ]
    end_mask = df["type_name"].isin(end_list)
    if end_mask.any():
        end_noise = rng.normal(0.0, sigma, size=df.loc[end_mask, ["end_x", "end_y"]].shape)
        df.loc[end_mask, ["end_x", "end_y"]] += end_noise

    return df


def _add_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach possession id, index in possession, and possession length."""
    out = df.copy()
    if out.empty:
        out["poss_id"] = pd.Series(dtype=int)
        out["idx_in_seq"] = pd.Series(dtype=int)
        out["seq_length"] = pd.Series(dtype=int)
        return out

    out["poss_id"] = out["new_poss"].cumsum().astype(int)
    out["idx_in_seq"] = out.groupby("poss_id").cumcount().astype(int)
    out["seq_length"] = out.groupby("poss_id")["poss_id"].transform("size").astype(int)
    return out


def prepare_data(
    actions: pd.DataFrame,
    action_map: Mapping[str, int],
    *,
    min_sequence_length: int = 3,
    sigma: float = 0.5,
    random_state: int | None = None,
) -> tuple[Array, Array]:
    """
    Build model features and sequence lengths from action-level SPADL data.

    Returns
    -------
    X : Array
        Feature matrix with columns:
        [action_type_id, start_x, start_y, cos_angle, sin_angle]
    lengths : Array
        Possession sequence lengths in order of poss_id.
    """
    if min_sequence_length < 1:
        raise ValueError("min_sequence_length must be >= 1.")
    _require_columns(actions, _BASE_REQUIRED_COLUMNS)
    if not action_map:
        return _empty_features()

    df = consolidate(actions)
    df = add_noise(df, sigma=sigma, random_state=random_state)

    dx = df["end_x"] - df["start_x"]
    dy = df["end_y"] - df["start_y"]
    df["angle"] = np.arctan2(dy, dx)
    df["cos_angle"] = np.cos(df["angle"])
    df["sin_angle"] = np.sin(df["angle"])
    df["action_type_id"] = df["type_name"].map(action_map)

    df["new_poss"] = _compute_new_possession_mask(df)
    df = _add_sequence_columns(df)

    # Drop any whole sequence (poss_id) that contains at least one clearance.
    has_clearance = (
        df["type_name"].eq("clearance")
        .groupby(df["poss_id"])
        .transform("any")
    )
    df = df.loc[~has_clearance].copy()
    if df.empty:
        return _empty_features()

    valid_action = df["type_name"].isin(action_map)
    long_enough_seq = df["seq_length"] >= min_sequence_length
    df = df.loc[valid_action & long_enough_seq].copy()
    if df.empty:
        return _empty_features()

    df["new_poss"] = _compute_new_possession_mask(df)
    df = _add_sequence_columns(df)
    df = df.loc[df["seq_length"] >= min_sequence_length].copy()
    if df.empty:
        return _empty_features()

    x = df[_FEATURE_COLUMNS].to_numpy(dtype=float)
    lengths = df.groupby("poss_id").size().to_numpy(dtype=int)
    return x, lengths
