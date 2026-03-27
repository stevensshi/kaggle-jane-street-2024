"""Phase 3: Baseline Models for Jane Street 2024.

Memory-optimized for 15GB RAM: column-by-column numpy conversion to avoid
polars copy overhead. Training window: dates 900-1188 (~10M rows).
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

TRAIN_PATH = str(Path(__file__).resolve().parent.parent / "data" / "raw" / "train.parquet")
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "baselines"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [f"feature_{i:02d}" for i in range(79)]
TARGET_COL = "responder_6"
TRAIN_LO, TRAIN_HI = 900, 1188
VAL_LO, VAL_HI = 1189, 1443

results = {}


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}", flush=True)


def load_Xyw(date_lo, date_hi, feat_cols):
    """Load date range into (X, y, w, dates) numpy arrays, column-by-column."""
    all_cols = list(set(["date_id", "weight", TARGET_COL] + feat_cols))
    df = (
        pl.scan_parquet(TRAIN_PATH)
        .filter((pl.col("date_id") >= date_lo) & (pl.col("date_id") <= date_hi))
        .select(all_cols)
        .collect(engine="streaming")
    )
    n = len(df)
    nf = len(feat_cols)

    # Pre-allocate and fill column-by-column (avoids full DataFrame copy)
    X = np.empty((n, nf), dtype=np.float32)
    for i, c in enumerate(feat_cols):
        col = df[c].to_numpy()
        np.nan_to_num(col, copy=False)
        X[:, i] = col
        del col

    y = df[TARGET_COL].to_numpy().astype(np.float32)
    w = df["weight"].to_numpy().astype(np.float32)
    dates = df["date_id"].to_numpy()
    del df
    gc.collect()
    return X, y, w, dates


# ─────────────────────────────────────────────────────────────
section("Loading validation data")
t0 = time.time()
X_val, y_val, w_val, _ = load_Xyw(VAL_LO, VAL_HI, FEATURE_COLS)
print(f"Val: {X_val.shape[0]:,} rows in {time.time()-t0:.1f}s  (y: mean={y_val.mean():.5f} std={y_val.std():.4f})")

# ─────────────────────────────────────────────────────────────
section("3.1a: Predict Zero")
r2_zero = weighted_r2(y_val, np.zeros_like(y_val), w_val)
print(f"R² = {r2_zero:.6f}")
results["predict_zero"] = {"r2": float(r2_zero)}

# ─────────────────────────────────────────────────────────────
section("3.1b: Predict lag_1 responder_6")
print("Constructing lag from train tail + val...")
lag_df = (
    pl.scan_parquet(TRAIN_PATH)
    .filter(pl.col("date_id") >= TRAIN_HI - 5)
    .filter(pl.col("date_id") <= VAL_HI)
    .select(["date_id", "time_id", "symbol_id", "weight", TARGET_COL])
    .collect(engine="streaming")
    .sort(["symbol_id", "date_id", "time_id"])
)
lag_df = lag_df.with_columns(pl.col(TARGET_COL).shift(1).over("symbol_id").alias("lag1"))
vl = lag_df.filter(pl.col("date_id") >= VAL_LO).drop_nulls(subset=["lag1"])
y_vl = vl[TARGET_COL].to_numpy().astype(np.float64)
w_vl = vl["weight"].to_numpy().astype(np.float64)
p_l1 = vl["lag1"].to_numpy().astype(np.float64)

r2_l1r = weighted_r2(y_vl, p_l1, w_vl)
sc = np.sum(w_vl * y_vl * p_l1) / np.sum(w_vl * p_l1**2)
r2_l1s = weighted_r2(y_vl, sc * p_l1, w_vl)
print(f"lag_1 raw:              R² = {r2_l1r:.6f}")
print(f"lag_1 scaled (s={sc:.4f}): R² = {r2_l1s:.6f}")
results["predict_lag1"] = {"r2_raw": float(r2_l1r), "r2_scaled": float(r2_l1s), "scale": float(sc)}
del lag_df, vl, y_vl, w_vl, p_l1; gc.collect()

# ─────────────────────────────────────────────────────────────
section(f"Loading training data (dates {TRAIN_LO}-{TRAIN_HI})")
t0 = time.time()
X_train, y_train, w_train, train_dates = load_Xyw(TRAIN_LO, TRAIN_HI, FEATURE_COLS)
print(f"Train: {X_train.shape[0]:,} rows in {time.time()-t0:.1f}s  (y: mean={y_train.mean():.5f} std={y_train.std():.4f})")

# ─────────────────────────────────────────────────────────────
section("3.1c: Ridge Regression")
from sklearn.linear_model import Ridge

top5 = ["feature_06", "feature_04", "feature_07", "feature_36", "feature_60"]
top5_i = [FEATURE_COLS.index(f) for f in top5]

t0 = time.time()
rdg = Ridge(alpha=1.0).fit(X_train[:, top5_i], y_train, sample_weight=w_train)
tt5 = time.time() - t0
r2_r5 = weighted_r2(y_val, rdg.predict(X_val[:, top5_i]), w_val)
print(f"Ridge (top 5):  R² = {r2_r5:.6f}  train={tt5:.2f}s")
print(f"  Coefs: {dict(zip(top5, rdg.coef_.round(4)))}")
results["ridge_top5"] = {"r2": float(r2_r5), "features": top5, "train_time": tt5}

t0 = time.time()
rdg79 = Ridge(alpha=1.0).fit(X_train, y_train, sample_weight=w_train)
tt79 = time.time() - t0
r2_r79 = weighted_r2(y_val, rdg79.predict(X_val), w_val)
print(f"Ridge (all 79): R² = {r2_r79:.6f}  train={tt79:.2f}s")
results["ridge_all79"] = {"r2": float(r2_r79), "train_time": tt79}
del rdg, rdg79; gc.collect()

# ─────────────────────────────────────────────────────────────
section("3.2: LightGBM (all 79 features)")
import lightgbm as lgb

lgb_p = {
    "objective": "regression", "metric": "mse",
    "learning_rate": 0.05, "num_leaves": 127, "max_depth": -1,
    "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "n_jobs": -1, "verbose": -1, "seed": 42,
}

lgb_tr = lgb.Dataset(X_train, y_train, weight=w_train, feature_name=FEATURE_COLS, free_raw_data=False)
lgb_vl = lgb.Dataset(X_val, y_val, weight=w_val, reference=lgb_tr, feature_name=FEATURE_COLS, free_raw_data=False)

t0 = time.time()
m_lgb = lgb.train(
    lgb_p, lgb_tr, num_boost_round=2000,
    valid_sets=[lgb_tr, lgb_vl], valid_names=["train", "val"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
)
tt_lgb = time.time() - t0

r2_lgb = weighted_r2(y_val, m_lgb.predict(X_val), w_val)
r2_lgb_tr = weighted_r2(y_train, m_lgb.predict(X_train), w_train)

t0 = time.time()
for _ in range(100): m_lgb.predict(X_val[:1])
inf_lgb = (time.time() - t0) / 100 * 1000

print(f"Val R² = {r2_lgb:.6f}, Train R² = {r2_lgb_tr:.6f}, gap = {r2_lgb_tr-r2_lgb:.6f}")
print(f"Best iter: {m_lgb.best_iteration}, train: {tt_lgb:.1f}s, infer: {inf_lgb:.3f}ms/row")
results["lgb_all79"] = {
    "r2": float(r2_lgb), "r2_train": float(r2_lgb_tr),
    "best_iteration": m_lgb.best_iteration, "train_time": tt_lgb, "infer_ms": inf_lgb,
}
del lgb_tr, lgb_vl; gc.collect()

# ─────────────────────────────────────────────────────────────
section("3.3: Feature Selection")

ig = m_lgb.feature_importance(importance_type="gain").astype(np.float64)
isp = m_lgb.feature_importance(importance_type="split")
order = np.argsort(ig)[::-1]
total = ig.sum()

print("\nTop 20 by gain:")
for i in range(20):
    j = order[i]
    print(f"  {FEATURE_COLS[j]:>12}  gain={ig[j]:10.1f} ({ig[j]/total*100:5.1f}%)  splits={isp[j]:6d}")

cg = np.cumsum(ig[order]) / total
for thr in [0.5, 0.7, 0.8, 0.9, 0.95]:
    print(f"  {thr*100:.0f}% gain → top {int(np.searchsorted(cg, thr))+1}")

gps = ig / np.maximum(isp, 1).astype(np.float64)
print("\nLowest gain/split:")
for j in np.argsort(gps)[:10]:
    print(f"  {FEATURE_COLS[j]:>12}  gain/split={gps[j]:.2f}  splits={isp[j]}")

# Stability
print("\n--- Stability ---")
mid = (TRAIN_LO + TRAIN_HI) // 2
em, lm = train_dates <= mid, train_dates > mid
me = lgb.train(lgb_p, lgb.Dataset(X_train[em], y_train[em], weight=w_train[em]),
               num_boost_round=m_lgb.best_iteration)
ml = lgb.train(lgb_p, lgb.Dataset(X_train[lm], y_train[lm], weight=w_train[lm]),
               num_boost_round=m_lgb.best_iteration)
ie = me.feature_importance(importance_type="gain").astype(np.float64)
il = ml.feature_importance(importance_type="gain").astype(np.float64)

from scipy.stats import spearmanr
rc, _ = spearmanr(ie, il)
print(f"Spearman corr: {rc:.3f}")

t30e = set(np.array(FEATURE_COLS)[np.argsort(ie)[-30:]])
t30l = set(np.array(FEATURE_COLS)[np.argsort(il)[-30:]])
stable = sorted(t30e & t30l)
print(f"Stable (top-30 both): {len(stable)} — {stable}")
del me, ml; gc.collect()

# Select by 90% gain
ns = int(np.searchsorted(cg, 0.90)) + 1
sf = [FEATURE_COLS[order[i]] for i in range(ns)]
si = [FEATURE_COLS.index(f) for f in sf]
print(f"\nSelected {ns} features (90% gain): {sf}")

Xts = X_train[:, si].copy()
Xvs = X_val[:, si].copy()
# Free full 79-feature arrays — only need selected subset from here
del X_train, X_val; gc.collect()

lgb_ts = lgb.Dataset(Xts, y_train, weight=w_train, feature_name=sf, free_raw_data=False)
lgb_vs = lgb.Dataset(Xvs, y_val, weight=w_val, reference=lgb_ts, feature_name=sf, free_raw_data=False)

t0 = time.time()
m_lgbs = lgb.train(
    lgb_p, lgb_ts, num_boost_round=2000,
    valid_sets=[lgb_ts, lgb_vs], valid_names=["train", "val"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
)
tt_lgbs = time.time() - t0
r2_lgbs = weighted_r2(y_val, m_lgbs.predict(Xvs), w_val)
print(f"\nLGBM (top-{ns}): R² = {r2_lgbs:.6f} (vs all-79: {r2_lgb:.6f})")
print(f"Train: {tt_lgbs:.1f}s, best iter: {m_lgbs.best_iteration}")
results["lgb_selected"] = {
    "r2": float(r2_lgbs), "n_features": ns, "features": sf,
    "train_time": tt_lgbs, "best_iteration": m_lgbs.best_iteration,
}
results["feature_stability"] = {"rank_correlation": float(rc), "stable_features": stable}
del lgb_ts, lgb_vs; gc.collect()

# ─────────────────────────────────────────────────────────────
section("3.4: Model Comparison (selected features)")

# Ridge
t0 = time.time()
rs = Ridge(alpha=1.0).fit(Xts, y_train, sample_weight=w_train)
ttr = time.time() - t0
r2_rs = weighted_r2(y_val, rs.predict(Xvs), w_val)
t0 = time.time()
for _ in range(1000): rs.predict(Xvs[:1])
infr = (time.time() - t0) / 1000 * 1000
print(f"Ridge:    R² = {r2_rs:.6f}  train={ttr:.2f}s  infer={infr:.3f}ms")
del rs; gc.collect()

# XGBoost
import xgboost as xgb
xp = {
    "objective": "reg:squarederror", "learning_rate": 0.05,
    "max_depth": 7, "min_child_weight": 100,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "tree_method": "hist", "nthread": -1, "seed": 42,
}
dt = xgb.DMatrix(Xts, y_train, weight=w_train, feature_names=sf)
dv = xgb.DMatrix(Xvs, y_val, weight=w_val, feature_names=sf)
t0 = time.time()
mx = xgb.train(xp, dt, num_boost_round=2000,
               evals=[(dt, "train"), (dv, "val")],
               early_stopping_rounds=50, verbose_eval=200)
ttx = time.time() - t0
r2_x = weighted_r2(y_val, mx.predict(dv), w_val)
dv1 = xgb.DMatrix(Xvs[:1], feature_names=sf)
t0 = time.time()
for _ in range(100): mx.predict(dv1)
infx = (time.time() - t0) / 100 * 1000
print(f"XGBoost:  R² = {r2_x:.6f}  train={ttx:.1f}s  infer={infx:.3f}ms")
print(f"LightGBM: R² = {r2_lgbs:.6f}  train={tt_lgbs:.1f}s  infer={inf_lgb:.3f}ms")

results["model_comparison"] = {
    "ridge": {"r2": float(r2_rs), "train_time": ttr, "infer_ms": infr},
    "xgboost": {"r2": float(r2_x), "train_time": ttx, "infer_ms": infx, "best_iteration": mx.best_iteration},
    "lightgbm": {"r2": float(r2_lgbs), "train_time": tt_lgbs, "infer_ms": inf_lgb},
}

# ─────────────────────────────────────────────────────────────
section(f"SUMMARY (val dates {VAL_LO}-{VAL_HI})")
n_tr = Xts.shape[0]
n_va = Xvs.shape[0]

print(f"\n{'Model':<32} {'R²':>10} {'Train':>8} {'Infer':>10}")
print("-" * 64)
for name, r2, tt, inf in [
    ("Predict Zero",           r2_zero,  "-",          "-"),
    ("Predict lag_1 (raw)",    r2_l1r,   "-",          "-"),
    ("Predict lag_1 (scaled)", r2_l1s,   "-",          "-"),
    ("Ridge (top 5)",          r2_r5,    f"{tt5:.1f}s", "-"),
    ("Ridge (all 79)",         r2_r79,   f"{tt79:.1f}s","-"),
    (f"Ridge (sel {ns})",      r2_rs,    f"{ttr:.1f}s", f"{infr:.2f}ms"),
    ("LightGBM (all 79)",     r2_lgb,   f"{tt_lgb:.0f}s",f"{inf_lgb:.2f}ms"),
    (f"LightGBM (sel {ns})",  r2_lgbs,  f"{tt_lgbs:.0f}s",f"{inf_lgb:.2f}ms"),
    (f"XGBoost (sel {ns})",   r2_x,     f"{ttx:.0f}s", f"{infx:.2f}ms"),
]:
    print(f"{name:<32} {r2:>10.6f} {tt:>8} {inf:>10}")

print(f"\nTrain: dates {TRAIN_LO}-{TRAIN_HI} ({n_tr:,} rows)")
print(f"Val:   dates {VAL_LO}-{VAL_HI} ({n_va:,} rows)")

# Save
rp = REPORT_DIR / "results.json"
def cvt(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o
with open(rp, "w") as f:
    json.dump(results, f, indent=2, default=cvt)
print(f"\nSaved to {rp}")
