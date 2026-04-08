"""Phase 4: Feature Engineering for Jane Street 2024.

Memory design: Two-pass approach (train, then val) to avoid OOM.
  - Val pass loads a small overlap from training data for correct lag/rolling at the boundary.

New features added on top of the 54 selected features from Phase 3:
  1. Lag responder features: lag1_r6 (corr=0.90), lag1_r3 (0.76), lag1_r7 (0.42),
                             lag1_r4 (0.36), lag1_r0 (-0.09)
  2. Market features: cross-symbol mean + deviation for top 10 features (20 features)
  3. Symbol rolling features: 10-step rolling mean for top 10 features (10 features)

Total: 54 + 5 + 20 + 10 = 89 features

Phase 3 baseline: LightGBM (54 features, no lags) R² = 0.010782
"""

import gc
import time
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import weighted_r2

TRAIN_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "raw" / "train.parquet"
)
assert Path(TRAIN_PATH).exists(), f"Data not found: {TRAIN_PATH}"

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "feature_engineering"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 54 features selected in Phase 3 (90% cumulative LightGBM gain, ordered by gain)
SELECTED_54 = [
    "feature_06", "feature_61", "feature_30", "feature_36", "feature_07",
    "feature_04", "feature_24", "feature_31", "feature_29", "feature_20",
    "feature_22", "feature_21", "feature_58", "feature_08", "feature_47",
    "feature_01", "feature_25", "feature_28", "feature_05", "feature_60",
    "feature_26", "feature_23", "feature_38", "feature_72", "feature_69",
    "feature_27", "feature_59", "feature_33", "feature_52", "feature_15",
    "feature_49", "feature_66", "feature_56", "feature_14", "feature_37",
    "feature_34", "feature_17", "feature_42", "feature_70", "feature_77",
    "feature_45", "feature_13", "feature_12", "feature_53", "feature_67",
    "feature_78", "feature_11", "feature_57", "feature_19", "feature_65",
    "feature_10", "feature_16", "feature_09", "feature_18",
]

# Top 10 features by LightGBM gain (for market + rolling)
TOP10 = SELECTED_54[:10]

# Responders for lag features (by |corr| with responder_6: 0.90, 0.76, 0.42, 0.36, -0.09)
LAG_RESPONDERS = [6, 3, 7, 4, 0]

TARGET_COL = "responder_6"
TRAIN_LO, TRAIN_HI = 900, 1188
VAL_LO, VAL_HI = 1189, 1443
# Overlap: load this many days of training data before val to get correct lag/rolling at boundary
VAL_OVERLAP_START = 1174  # 15 days before val start

PHASE3_BASELINE_R2 = 0.010782
LAG_BASELINE_R2 = 0.811445

LOAD_COLS = list(set(
    ["date_id", "time_id", "symbol_id", "weight", TARGET_COL]
    + SELECTED_54
    + [f"responder_{i}" for i in LAG_RESPONDERS]
))

LAG_FEAT_COLS = [f"lag1_r{i}" for i in LAG_RESPONDERS]
MKT_FEAT_COLS = [f"mkt_mean_{f}" for f in TOP10] + [f"rel_{f}" for f in TOP10]
ROLL_FEAT_COLS = [f"roll10_{f}" for f in TOP10]
ALL_FEATURES = SELECTED_54 + LAG_FEAT_COLS + MKT_FEAT_COLS + ROLL_FEAT_COLS  # 89 total

results = {}


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}", flush=True)


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute all engineered features in-place on a sorted DataFrame.

    Input df must be sorted by (symbol_id, date_id, time_id).
    Market features use all rows in df (so pass train-only or val-only data
    for market aggregation, but pass overlap+val for correct lag/rolling).
    """
    # 4.1 Lag responder features
    df = df.with_columns([
        pl.col(f"responder_{i}").shift(1).over("symbol_id").alias(f"lag1_r{i}")
        for i in LAG_RESPONDERS
    ])

    # 4.3 Rolling mean (window=10, within each symbol)
    df = df.with_columns([
        pl.col(f).rolling_mean(window_size=10, min_samples=1).over("symbol_id").alias(f"roll10_{f}")
        for f in TOP10
    ])

    # Drop responder columns used for lags — keep responder_6 (it's the target)
    drop_responders = [f"responder_{i}" for i in LAG_RESPONDERS if i != 6]
    df = df.drop(drop_responders)

    # 4.2 Market features: cross-symbol mean per time slot
    mkt_df = df.group_by(["date_id", "time_id"]).agg([
        pl.col(f).mean().alias(f"mkt_mean_{f}") for f in TOP10
    ])
    df = df.join(mkt_df, on=["date_id", "time_id"], how="left")
    del mkt_df

    # Relative deviation: feature - market mean
    df = df.with_columns([
        (pl.col(f) - pl.col(f"mkt_mean_{f}")).alias(f"rel_{f}")
        for f in TOP10
    ])

    return df


def extract_numpy(df: pl.DataFrame, date_lo: int, date_hi: int):
    """Extract (X, y, w) numpy arrays from df, filtered to [date_lo, date_hi].
    Fills NaN with 0 during extraction (column-by-column to minimize peak memory).
    """
    split_df = df.filter(
        (pl.col("date_id") >= date_lo) & (pl.col("date_id") <= date_hi)
    )
    n = len(split_df)
    nf = len(ALL_FEATURES)

    X = np.empty((n, nf), dtype=np.float32)
    for i, col in enumerate(ALL_FEATURES):
        arr = split_df[col].to_numpy()
        np.nan_to_num(arr, copy=False)
        X[:, i] = arr
        del arr

    y = split_df[TARGET_COL].to_numpy().astype(np.float32)
    w = split_df["weight"].to_numpy().astype(np.float32)
    del split_df
    return X, y, w


# ─────────────────────────────────────────────────────────────
section("Pass 1: Training data (dates 900–1188)")
t0 = time.time()

train_raw = (
    pl.scan_parquet(TRAIN_PATH)
    .filter((pl.col("date_id") >= TRAIN_LO) & (pl.col("date_id") <= TRAIN_HI))
    .select(LOAD_COLS)
    .collect(engine="streaming")
    .sort(["symbol_id", "date_id", "time_id"])
)
print(f"Loaded {len(train_raw):,} rows in {time.time()-t0:.1f}s")

t0 = time.time()
train_eng = engineer_features(train_raw)
del train_raw; gc.collect()
print(f"Features engineered in {time.time()-t0:.1f}s")

t0 = time.time()
X_train, y_train, w_train = extract_numpy(train_eng, TRAIN_LO, TRAIN_HI)
del train_eng; gc.collect()
print(f"Train arrays: X={X_train.shape}, extracted in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────
section("Pass 2: Validation data (dates 1189–1443, with overlap for lag/rolling)")
t0 = time.time()

val_raw = (
    pl.scan_parquet(TRAIN_PATH)
    .filter((pl.col("date_id") >= VAL_OVERLAP_START) & (pl.col("date_id") <= VAL_HI))
    .select(LOAD_COLS)
    .collect(engine="streaming")
    .sort(["symbol_id", "date_id", "time_id"])
)
print(f"Loaded {len(val_raw):,} rows in {time.time()-t0:.1f}s")

t0 = time.time()
val_eng = engineer_features(val_raw)
del val_raw; gc.collect()
print(f"Features engineered in {time.time()-t0:.1f}s")

t0 = time.time()
X_val, y_val, w_val = extract_numpy(val_eng, VAL_LO, VAL_HI)
del val_eng; gc.collect()
print(f"Val arrays: X={X_val.shape}, extracted in {time.time()-t0:.1f}s")

print(f"\nTotal features: {len(ALL_FEATURES)}")
print(f"  - Base (54 selected): {len(SELECTED_54)}")
print(f"  - Lag responders:     {len(LAG_FEAT_COLS)}")
print(f"  - Market features:    {len(MKT_FEAT_COLS)}")
print(f"  - Rolling features:   {len(ROLL_FEAT_COLS)}")


# ─────────────────────────────────────────────────────────────
import lightgbm as lgb

LGB_PARAMS = {
    "objective": "regression", "metric": "mse",
    "learning_rate": 0.05, "num_leaves": 127, "max_depth": -1,
    "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "n_jobs": -1, "verbose": -1, "seed": 42,
}


def train_lgb(X_tr, X_va, y_tr, y_va, w_tr, w_va, feat_names, label):
    ds_tr = lgb.Dataset(X_tr, y_tr, weight=w_tr, feature_name=feat_names, free_raw_data=False)
    ds_va = lgb.Dataset(X_va, y_va, weight=w_va, reference=ds_tr, feature_name=feat_names, free_raw_data=False)
    t0 = time.time()
    m = lgb.train(
        LGB_PARAMS, ds_tr, num_boost_round=2000,
        valid_sets=[ds_tr, ds_va], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )
    tt = time.time() - t0
    r2_va = weighted_r2(y_va, m.predict(X_va), w_va)
    r2_tr = weighted_r2(y_tr, m.predict(X_tr), w_tr)
    t0 = time.time()
    for _ in range(100): m.predict(X_va[:1])
    infer_ms = (time.time() - t0) / 100 * 1000
    print(f"\n{label}:")
    print(f"  Val R²   = {r2_va:.6f}   (Phase 3 baseline: {PHASE3_BASELINE_R2:.6f})")
    print(f"  Train R² = {r2_tr:.6f}   (overfit gap: {r2_tr - r2_va:.6f})")
    print(f"  Best iter: {m.best_iteration}, train: {tt:.0f}s, infer: {infer_ms:.3f}ms/row")
    return m, r2_va, r2_tr, tt, infer_ms


# ─────────────────────────────────────────────────────────────
section("Ablation A: 54 selected + 5 lag features (key question: how much does lag help?)")

lag_feats = SELECTED_54 + LAG_FEAT_COLS
lag_idx = [ALL_FEATURES.index(f) for f in lag_feats]

m_lag, r2_lag, r2_lag_tr, tt_lag, infer_lag = train_lgb(
    X_train[:, lag_idx], X_val[:, lag_idx],
    y_train, y_val, w_train, w_val,
    lag_feats, "LightGBM (54 sel + 5 lag)",
)
results["lgb_54_plus_lag"] = {
    "r2": float(r2_lag), "r2_train": float(r2_lag_tr),
    "best_iteration": m_lag.best_iteration,
    "train_time": tt_lag, "infer_ms": infer_lag,
    "n_features": len(lag_feats),
}
del m_lag; gc.collect()


# ─────────────────────────────────────────────────────────────
section("Full: 54 selected + 5 lag + 20 market + 10 rolling (89 features)")

m_full, r2_full, r2_full_tr, tt_full, infer_full = train_lgb(
    X_train, X_val,
    y_train, y_val, w_train, w_val,
    ALL_FEATURES, "LightGBM (89 engineered features)",
)

# Feature importance
ig = m_full.feature_importance(importance_type="gain").astype(np.float64)
order = np.argsort(ig)[::-1]
total = ig.sum()
print("\nTop 20 features by gain:")
for i in range(20):
    j = order[i]
    print(f"  {ALL_FEATURES[j]:>30}  {ig[j]/total*100:5.1f}%")

results["lgb_89_engineered"] = {
    "r2": float(r2_full), "r2_train": float(r2_full_tr),
    "best_iteration": m_full.best_iteration,
    "train_time": tt_full, "infer_ms": infer_full,
    "n_features": len(ALL_FEATURES),
    "features": ALL_FEATURES,
    "top20_by_gain": [
        {"feature": ALL_FEATURES[order[i]], "gain_pct": float(ig[order[i]] / total * 100)}
        for i in range(20)
    ],
}
del m_full; gc.collect()


# ─────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"\n{'Model':<55} {'Val R²':>10}")
print("-" * 67)
rows = [
    ("Phase 3: LightGBM (54 features, no lag)",         PHASE3_BASELINE_R2),
    ("Naive baseline: predict lag_1 × 0.90",             LAG_BASELINE_R2),
    ("LightGBM (54 sel + 5 lag)",                        r2_lag),
    ("LightGBM (89 = 54 sel + lag + market + rolling)",  r2_full),
]
for name, r2 in rows:
    print(f"  {name:<53} {r2:>10.6f}")

# Save results
rp = REPORT_DIR / "results.json"
def cvt(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o

with open(rp, "w") as f:
    json.dump(results, f, indent=2, default=cvt)
print(f"\nResults saved to {rp}")
