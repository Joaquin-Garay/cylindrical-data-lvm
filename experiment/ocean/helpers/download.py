"""Download and parse NOAA/NDBC historical standard meteorological files."""

import gzip
from io import StringIO
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

STATION = "46042"
YEARS = range(2000, 2023)
NDBC_STD_MET_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz"

COLUMN_ALIASES = {
    "YY": "year",
    "YYYY": "year",
    "MM": "month",
    "DD": "day",
    "hh": "hour",
    "mm": "minute",
    "WD": "wdir",
    "WDIR": "wdir",
    "WSPD": "wspd",
    "GST": "gst",
    "WVHT": "wvht",
    "DPD": "dpd",
    "APD": "apd",
    "MWD": "mwd",
    "BAR": "pres",
    "PRES": "pres",
    "ATMP": "atmp",
    "WTMP": "wtmp",
    "DEWP": "dewp",
    "VIS": "vis",
    "TIDE": "tide",
}

DATE_COLUMNS = ["year", "month", "day", "hour", "minute"]
METADATA_COLUMNS = {"station", "file_year", "time", *DATE_COLUMNS}
TEXT_MISSING_VALUES = {"": np.nan, "-": np.nan, "MM": np.nan, "N/A": np.nan}

# Historical NDBC missing codes are column-specific, so avoid one global 9-code rule.
COLUMN_MISSING_PATTERNS = {
    "wdir": r"^-?9{3}(?:\.0+)?$",
    "wspd": r"^-?9{2}(?:\.0+)?$",
    "gst": r"^-?9{2}(?:\.0+)?$",
    "wvht": r"^-?9{2}(?:\.0+)?$",
    "dpd": r"^-?9{2}(?:\.0+)?$",
    "apd": r"^-?9{2}(?:\.0+)?$",
    "mwd": r"^-?9{3}(?:\.0+)?$",
    "pres": r"^-?9{4}(?:\.0+)?$",
    "atmp": r"^-?9{3}(?:\.0+)?$",
    "wtmp": r"^-?9{3}(?:\.0+)?$",
    "dewp": r"^-?9{3}(?:\.0+)?$",
    "vis": r"^-?9{2}(?:\.0+)?$",
    "tide": r"^-?9{2}(?:\.0+)?$",
    "ptdy": r"^-?9{2}(?:\.0+)?$",
}


def find_project_root(start=None):
    start = Path(__file__).resolve() if start is None else Path(start).resolve()
    if start.is_file():
        start = start.parent

    for path in [start] + list(start.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path

    return start


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "ndbc"


def build_ndbc_stdmet_url(station, year):
    return NDBC_STD_MET_URL.format(station=station, year=year)


def download_ndbc_stdmet_file(station, year, data_dir=DATA_DIR, overwrite=False):
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{station}h{year}.txt.gz"
    if overwrite or not path.exists():
        urlretrieve(build_ndbc_stdmet_url(station, year), path)
    return path


def normalize_header(header_line):
    raw_columns = header_line.lstrip("#").split()
    return [COLUMN_ALIASES.get(column, column.lower()) for column in raw_columns]


def read_ndbc_stdmet_file(path, station=STATION, file_year=None):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    header_line = next(
        (line for line in lines if line.startswith("#YY") or line.startswith("YYYY")),
        None,
    )
    if header_line is None:
        raise ValueError(f"No NDBC header found in {path}")

    data_lines = [line for line in lines if line[0].isdigit()]
    if not data_lines:
        raise ValueError(f"No NDBC data rows found in {path}")

    frame = pd.read_csv(
        StringIO("\n".join(data_lines)),
        sep=r"\s+",
        names=normalize_header(header_line),
        dtype="string",
        engine="python",
    )

    if "minute" not in frame.columns:
        frame.insert(frame.columns.get_loc("hour") + 1, "minute", "0")

    for column in DATE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")

    time = pd.to_datetime(
        {
            "year": frame["year"],
            "month": frame["month"],
            "day": frame["day"],
            "hour": frame["hour"],
            "minute": frame["minute"],
        },
        errors="coerce",
        utc=True,
    )

    frame.insert(0, "station", station)
    frame.insert(1, "file_year", int(file_year) if file_year is not None else None)
    frame.insert(2, "time", time)

    measurement_columns = [column for column in frame.columns if column not in METADATA_COLUMNS]
    frame[measurement_columns] = frame[measurement_columns].replace(TEXT_MISSING_VALUES)
    for column, missing_pattern in COLUMN_MISSING_PATTERNS.items():
        if column in frame.columns:
            frame[column] = frame[column].replace(missing_pattern, np.nan, regex=True)
    frame[measurement_columns] = frame[measurement_columns].apply(pd.to_numeric, errors="coerce")

    return frame
