"""Data loading utilities for Jane Street 2024 competition.

Splits (from EDA Phase 2.3):
  Train:   date_id 0-1188  (1189 days)
  Val:     date_id 1189-1443 (255 days)
  Holdout: date_id 1444-1698 (255 days)
"""

import polars as pl
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TRAIN_PATH = DATA_DIR / "train.parquet"
LAGS_PATH = DATA_DIR / "lags.parquet"

# Temporal splits
TRAIN_END = 1188
VAL_START = 1189
VAL_END = 1443
HOLDOUT_START = 1444
HOLDOUT_END = 1698

FEATURE_COLS = [f"feature_{i:02d}" for i in range(79)]
RESPONDER_COLS = [f"responder_{i}" for i in range(9)]
TARGET_COL = "responder_6"


def load_train_val(
    columns: list[str] | None = None,
    fillna: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load train and validation splits.

    Args:
        columns: If provided, only load these columns (plus date_id, weight, target).
        fillna: If True, fill NaN with 0 in feature columns.

    Returns:
        (train_df, val_df) as polars DataFrames.
    """
    required = ["date_id", "time_id", "symbol_id", "weight", TARGET_COL]
    if columns is not None:
        load_cols = list(set(required + columns))
    else:
        load_cols = None

    scan = pl.scan_parquet(TRAIN_PATH)
    if load_cols is not None:
        scan = scan.select(load_cols)

    df = scan.filter(pl.col("date_id") <= VAL_END).collect()

    train = df.filter(pl.col("date_id") <= TRAIN_END)
    val = df.filter(pl.col("date_id") >= VAL_START)

    if fillna:
        feat_cols_present = [c for c in train.columns if c.startswith("feature_")]
        train = train.with_columns([pl.col(c).fill_null(0.0) for c in feat_cols_present])
        val = val.with_columns([pl.col(c).fill_null(0.0) for c in feat_cols_present])

    return train, val


def load_split(
    split: str,
    columns: list[str] | None = None,
    fillna: bool = True,
) -> pl.DataFrame:
    """Load a single split: 'train', 'val', or 'holdout'."""
    bounds = {
        "train": (0, TRAIN_END),
        "val": (VAL_START, VAL_END),
        "holdout": (HOLDOUT_START, HOLDOUT_END),
    }
    lo, hi = bounds[split]

    required = ["date_id", "time_id", "symbol_id", "weight", TARGET_COL]
    if columns is not None:
        load_cols = list(set(required + columns))
    else:
        load_cols = None

    scan = pl.scan_parquet(TRAIN_PATH)
    if load_cols is not None:
        scan = scan.select(load_cols)

    df = scan.filter(
        (pl.col("date_id") >= lo) & (pl.col("date_id") <= hi)
    ).collect()

    if fillna:
        feat_cols_present = [c for c in df.columns if c.startswith("feature_")]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in feat_cols_present])

    return df


def to_numpy(
    df: pl.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X, y, w) numpy arrays from a polars DataFrame."""
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df[TARGET_COL].to_numpy().astype(np.float64)
    w = df["weight"].to_numpy().astype(np.float64)
    return X, y, w
