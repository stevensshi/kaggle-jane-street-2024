"""Data loading with provenance manifests.

Every load_* function returns (DataFrame, LoadManifest).
No hard-coded date cutoffs; all ranges come from splits.py.

HOLDOUT and FOLDS_B loaders raise by default. Set the environment variable
JS_PHASE_UNLOCK to the appropriate token to enable them:
  JS_PHASE_UNLOCK=FOLDS_B_PHASE6   → enables load_folds_b_val()
  JS_PHASE_UNLOCK=HOLDOUT_PHASE6   → enables load_holdout()
  JS_PHASE_UNLOCK=RESERVE          → enables load_reserve() (future cycle)

This makes accidental reads loud without requiring filesystem permissions.
"""

import os
from pathlib import Path

import polars as pl

from provenance import LoadManifest, make_manifest
from splits import (
    TRAIN_RANGE, VAL_RANGE, HOLDOUT_RANGE, RESERVE_RANGE,
    FOLDS_A, FOLDS_B, inner_split,
    FEATURE_COLS, RESPONDER_COLS, LAG_COLS, TARGET_COL,
)

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR   = _ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.parquet"
LAGS_PATH  = DATA_DIR / "lags.parquet"   # inference template only (39 rows)


def _load_range(lo: int, hi: int,
                columns: list[str] | None = None,
                fillna: bool = True) -> pl.DataFrame:
    """Core loader: filter by date range, optionally select columns."""
    scan = pl.scan_parquet(str(TRAIN_PATH / "**" / "*.parquet"))
    scan = scan.filter((pl.col("date_id") >= lo) & (pl.col("date_id") <= hi))
    if columns is not None:
        needed = list({"date_id", "time_id", "symbol_id", "weight",
                       TARGET_COL} | set(columns))
        scan = scan.select([c for c in needed if c in
                            pl.scan_parquet(str(TRAIN_PATH / "**" / "*.parquet"))
                            .columns])
    df = scan.collect()
    if fillna:
        feat_present = [c for c in df.columns if c.startswith("feature_")]
        if feat_present:
            df = df.with_columns(
                [pl.col(c).fill_null(0.0) for c in feat_present]
            )
    return df


# ── Public loaders ───────────────────────────────────────────────────────────

def load_train(columns: list[str] | None = None,
               fillna: bool = True) -> tuple[pl.DataFrame, LoadManifest]:
    lo, hi = TRAIN_RANGE
    df = _load_range(lo, hi, columns, fillna)
    return df, make_manifest(df, lo, hi, "train")


def load_val(columns: list[str] | None = None,
             fillna: bool = True) -> tuple[pl.DataFrame, LoadManifest]:
    lo, hi = VAL_RANGE
    df = _load_range(lo, hi, columns, fillna)
    return df, make_manifest(df, lo, hi, "val")


def load_train_val(columns: list[str] | None = None,
                   fillna: bool = True
                   ) -> tuple[pl.DataFrame, pl.DataFrame,
                               LoadManifest, LoadManifest]:
    """Load train and val splits together (single scan, then filter)."""
    lo_tr, hi_tr = TRAIN_RANGE
    lo_v,  hi_v  = VAL_RANGE
    scan = pl.scan_parquet(str(TRAIN_PATH / "**" / "*.parquet"))
    scan = scan.filter(pl.col("date_id") <= hi_v)
    if columns is not None:
        needed = list({"date_id", "time_id", "symbol_id", "weight",
                       TARGET_COL} | set(columns))
        all_cols = pl.scan_parquet(str(TRAIN_PATH / "**" / "*.parquet")).columns
        scan = scan.select([c for c in needed if c in all_cols])
    df = scan.collect()
    if fillna:
        feat_present = [c for c in df.columns if c.startswith("feature_")]
        if feat_present:
            df = df.with_columns(
                [pl.col(c).fill_null(0.0) for c in feat_present]
            )
    train = df.filter(pl.col("date_id") <= hi_tr)
    val   = df.filter(pl.col("date_id") >= lo_v)
    return (train, val,
            make_manifest(train, lo_tr, hi_tr, "train"),
            make_manifest(val,   lo_v,  hi_v,  "val"))


def load_fold(fold_idx: int, pool: str = "A",
              columns: list[str] | None = None,
              fillna: bool = True
              ) -> tuple[pl.DataFrame, pl.DataFrame,
                          LoadManifest, LoadManifest]:
    """Load a single fold from FOLDS_A or FOLDS_B.

    Returns (train_df, val_df, train_manifest, val_manifest).
    The fold's val_range is returned for evaluation only — it must never be
    used during training or early stopping (use inner_split for that).
    """
    folds = FOLDS_A if pool == "A" else FOLDS_B
    if fold_idx >= len(folds):
        raise ValueError(f"Fold {fold_idx} out of range for FOLDS_{pool}")
    fold = folds[fold_idx]
    lo_tr, hi_tr = fold["train"]
    lo_v,  hi_v  = fold["val"]

    if pool == "B":
        _check_folds_b_unlock()

    tr = _load_range(lo_tr, hi_tr, columns, fillna)
    vl = _load_range(lo_v,  hi_v,  columns, fillna)
    return (tr, vl,
            make_manifest(tr, lo_tr, hi_tr, f"folds_{pool}[{fold_idx}].train"),
            make_manifest(vl, lo_v,  hi_v,  f"folds_{pool}[{fold_idx}].val"))


def load_folds_b_val(columns: list[str] | None = None,
                     fillna: bool = True
                     ) -> tuple[pl.DataFrame, LoadManifest]:
    """Load FOLDS_B validation slice. Requires JS_PHASE_UNLOCK=FOLDS_B_PHASE6."""
    _check_folds_b_unlock()
    lo, hi = FOLDS_B[0]["val"]
    df = _load_range(lo, hi, columns, fillna)
    return df, make_manifest(df, lo, hi, "folds_b_val")


def load_holdout(columns: list[str] | None = None,
                 fillna: bool = True) -> tuple[pl.DataFrame, LoadManifest]:
    """Load HOLDOUT slice. Requires JS_PHASE_UNLOCK=HOLDOUT_PHASE6."""
    token = os.environ.get("JS_PHASE_UNLOCK", "")
    if token != "HOLDOUT_PHASE6":
        raise RuntimeError(
            "load_holdout() requires JS_PHASE_UNLOCK=HOLDOUT_PHASE6. "
            "Set this only at Phase 6.3 after holdout_gate.yaml is committed."
        )
    lo, hi = HOLDOUT_RANGE
    df = _load_range(lo, hi, columns, fillna)
    return df, make_manifest(df, lo, hi, "holdout")


def load_reserve(columns: list[str] | None = None,
                 fillna: bool = True) -> tuple[pl.DataFrame, LoadManifest]:
    """Load RESERVE slice. Requires JS_PHASE_UNLOCK=RESERVE. Do not use this cycle."""
    token = os.environ.get("JS_PHASE_UNLOCK", "")
    if token != "RESERVE":
        raise RuntimeError(
            "load_reserve() is sealed for this research cycle. "
            "RESERVE (1572-1698) is the next-cycle gate only."
        )
    lo, hi = RESERVE_RANGE
    df = _load_range(lo, hi, columns, fillna)
    return df, make_manifest(df, lo, hi, "reserve")


# ── Helpers ──────────────────────────────────────────────────────────────────

def to_numpy(df: pl.DataFrame, feature_cols: list[str]):
    """Extract (X, y, w) numpy arrays from a DataFrame."""
    import numpy as np
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df[TARGET_COL].to_numpy().astype(np.float64)
    w = df["weight"].to_numpy().astype(np.float64)
    return X, y, w


def add_lag_responders(df: pl.DataFrame) -> pl.DataFrame:
    """Compute lag-1 responders within each (symbol_id, date_id) group.

    responder_X_lag_1 at (symbol, date, time) = responder_X at previous time_id.
    At time_id=0, uses the last slot of the previous date_id for same symbol.

    This is the canonical lag contract. The audit script verifies it.
    """
    from splits import RESPONDER_COLS
    # Sort to ensure correct shift order
    df = df.sort(["symbol_id", "date_id", "time_id"])
    for col in RESPONDER_COLS:
        lag_name = f"{col}_lag_1"
        df = df.with_columns(
            pl.col(col)
              .shift(1)
              .over(["symbol_id", "date_id"])
              .alias(lag_name + "_intraday")
        )
        # For time_id=0, fill from previous date's last slot
        prev_last = (
            df.filter(
                pl.col("time_id") == df.group_by(["symbol_id", "date_id"])
                                        .agg(pl.col("time_id").max())
                                        .rename({"time_id": "max_time_id"})
                                        .join(df.select(["symbol_id", "date_id"]),
                                              on=["symbol_id", "date_id"])["max_time_id"]
            )
        )
        # Simpler: use global shift over (symbol_id,) and mark day boundaries
        df = df.with_columns(
            pl.col(col)
              .shift(1)
              .over("symbol_id")
              .alias(lag_name + "_global")
        )
        # Use intraday shift; only fall back to global at time_id==0
        df = df.with_columns(
            pl.when(pl.col("time_id") == 0)
              .then(pl.col(lag_name + "_global"))
              .otherwise(pl.col(lag_name + "_intraday"))
              .alias(lag_name)
        ).drop([lag_name + "_intraday", lag_name + "_global"])
    return df


def _check_folds_b_unlock():
    token = os.environ.get("JS_PHASE_UNLOCK", "")
    if token != "FOLDS_B_PHASE6":
        raise RuntimeError(
            "FOLDS_B access requires JS_PHASE_UNLOCK=FOLDS_B_PHASE6. "
            "Set this only at Phase 6.1, after the full design is frozen on FOLDS_A."
        )
