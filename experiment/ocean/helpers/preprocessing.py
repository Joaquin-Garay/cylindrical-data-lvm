"""Preprocess the NDBC wave data used in the ocean experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

from .download import DATA_DIR


DEFAULT_LINEAR_FEATURES = ("wvht", "apd", "wspd")
DEFAULT_DIRECTION_FEATURE = "mwd"
DEFAULT_DIRECTION_FEATURE_NAMES = ("cos(mwd)", "sin(mwd)")
DEFAULT_CSV_PATH = DATA_DIR / "ocean_data.csv"
DEFAULT_IRREGULAR_CUTOFF = pd.Timestamp("2019-11-10 10:30:00+00:00")


def _direction_vectors(direction_degrees) -> np.ndarray:
    radians = np.deg2rad(np.asarray(direction_degrees, dtype=float))
    return np.column_stack((np.cos(radians), np.sin(radians)))


def prepare_data(
        csv_path: str | Path = DEFAULT_CSV_PATH,
        *,
        linear_features: Sequence[str] = DEFAULT_LINEAR_FEATURES,
        direction_feature: str = DEFAULT_DIRECTION_FEATURE,
        train_fraction: float = 0.8,
        shuffle: bool = False,
        random_state: int | None = 57,
        ) -> dict[str, Any]:
    """Read the ocean CSV and build train/test matrices for the fixed experiment."""
    linear_features = tuple(linear_features)
    feature_columns = [*linear_features, direction_feature]

    df = pd.read_csv(csv_path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    regular_minutes = (
        (df["time"] < DEFAULT_IRREGULAR_CUTOFF)
        | (df["minute"] == 40)
    )
    frame = df.loc[regular_minutes, ["time", *feature_columns]].copy()
    frame[feature_columns] = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=feature_columns).reset_index(drop=True)
    frame.insert(0, "source_row_id", np.arange(len(frame), dtype=int))

    train_frame, test_frame = train_test_split(
        frame,
        train_size=train_fraction,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    train_frame = train_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)

    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    x_train_linear_raw = train_frame.loc[:, linear_features].to_numpy(dtype=float)
    x_test_linear_raw = test_frame.loc[:, linear_features].to_numpy(dtype=float)
    x_train_linear = transformer.fit_transform(x_train_linear_raw)
    x_test_linear = transformer.transform(x_test_linear_raw)

    x_train_direction = _direction_vectors(train_frame[direction_feature])
    x_test_direction = _direction_vectors(test_frame[direction_feature])
    x_train = np.concatenate((x_train_linear, x_train_direction), axis=1)
    x_test = np.concatenate((x_test_linear, x_test_direction), axis=1)

    return {
        "frame": frame,
        "train_frame": train_frame,
        "test_frame": test_frame,
        "x_train": x_train,
        "x_test": x_test,
        "x_train_linear": x_train_linear,
        "x_test_linear": x_test_linear,
        "x_train_direction": x_train_direction,
        "x_test_direction": x_test_direction,
        "x_train_linear_raw": x_train_linear_raw,
        "x_test_linear_raw": x_test_linear_raw,
        "linear_features": linear_features,
        "direction_feature": direction_feature,
        "direction_feature_names": DEFAULT_DIRECTION_FEATURE_NAMES,
        "feature_names": (*linear_features, *DEFAULT_DIRECTION_FEATURE_NAMES),
        "d_gauss": len(linear_features),
        "d_vmf": len(DEFAULT_DIRECTION_FEATURE_NAMES),
        "transformer": transformer,
        "split_index": len(train_frame),
        "train_fraction": train_fraction,
        "shuffle": shuffle,
        "random_state": random_state,
        "train_time_range": (train_frame["time"].min(), train_frame["time"].max()),
        "test_time_range": (test_frame["time"].min(), test_frame["time"].max()),
        "train_row_ids": train_frame["source_row_id"].to_numpy(dtype=int),
        "test_row_ids": test_frame["source_row_id"].to_numpy(dtype=int),
    }
